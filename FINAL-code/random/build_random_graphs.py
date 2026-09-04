# -*- coding: utf-8 -*-
"""
Positive-only degree-preserving RandomGraph generator for ESWA Major Revision.

Purpose
-------
Construct multiple randomized placebo graphs from the Positive-only Real Graph-2 cache
WITHOUT rebuilding correlations from 5-minute returns.

The randomization operates on the actual directed KNN adjacency consumed by GraphMix.
For every trading day it preserves EXACTLY:
    1) every node's out-degree;
    2) every node's in-degree;
    3) total number of active directed edges;
    4) daily graph density;
    5) node-validity mask.

It destroys the identity of cross-stock links through directed double-edge swaps:
    (a -> b, c -> d)  ==>  (a -> d, c -> b)

while rejecting proposals that would create:
    - self-loops;
    - duplicate edges.

Why directed?
-------------
The current Graph-2 cache is stored/consumed as a row-wise KNN adjacency. This script
preserves the exact computational degree structure actually seen by GraphMix, even if
the cached adjacency is not perfectly reciprocal after union/re-TopK processing.

Workflow
--------
FIRST RUN:
    SANITY_ONLY = True
    -> randomizes a small number of non-empty days for one seed
    -> checks exact degree preservation, density, self-loops, duplicates, edge overlap.

AFTER SANITY PASSES:
    SANITY_ONLY = False
    GRAPH_SEEDS = list(range(20))
    -> generates 20 independently randomized full graph caches.

Requires:
    numpy, pandas, numba
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception as e:
    raise ImportError(
        "This script requires numba. Install with: pip install numba"
    ) from e


# ============================================================
# 0) CONFIG
# ============================================================

# The script MUST use the Positive-only Real Graph-2 cache.
REAL_GRAPH_CACHE_CANDIDATES = [
    r"D:\PythonProject\LASSO_FINAL\graph_大修\graph2_positive_only\graphs"
    r"\roll20_binary_positive_mkt1_topk40_shift0_tau0.1_pow1.0_full1.npz",
]

OUT_DIR = Path(
    r"D:\PythonProject\LASSO_FINAL\graph_大修\positive_degree_preserving_random_graphs"
)

# ---- First run ----
SANITY_ONLY = False
SANITY_GRAPH_SEED = 0
SANITY_NONEMPTY_DAYS = 10

# ---- Full revision run ----
GRAPH_SEEDS = list(range(20))

# Number of SUCCESSFUL double-edge swaps targeted per active edge.
# 5.0 is already substantial; edge-overlap diagnostics are used to verify mixing.
SWAPS_PER_EDGE = 5.0

# Avoid an infinite loop if a graph is difficult to rewire.
MAX_ATTEMPTS_PER_TARGET_SWAP = 30

# After randomization, warn if too many original edges remain.
# This is not an identification "significance threshold"; it is only a mixing diagnostic.
WARN_OVERLAP_ABOVE = 0.10

# Hard structural checks.
REQUIRE_EXACT_OUT_DEGREE = True
REQUIRE_EXACT_IN_DEGREE = True
REQUIRE_ZERO_SELF_LOOPS = True
REQUIRE_ZERO_DUPLICATES = True


# ============================================================
# 1) HELPERS
# ============================================================

def resolve_real_graph_cache() -> Path:
    for p in REAL_GRAPH_CACHE_CANDIDATES:
        pp = Path(p)
        if pp.exists():
            return pp
    raise FileNotFoundError(
        "No Positive-only Real Graph-2 cache found. Checked:\n"
        + "\n".join(REAL_GRAPH_CACHE_CANDIDATES)
    )


def extract_active_directed_edges(
    nbr_idx_day: np.ndarray,
    nbr_w_day: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the exact active directed edges used by GraphMix:
        weight > 0 and target != source.
    Returns row-major edge arrays (u, v).
    """
    n, k = nbr_idx_day.shape
    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = nbr_idx_day.reshape(-1).astype(np.int32)
    weights = nbr_w_day.reshape(-1)

    mask = (
        np.isfinite(weights)
        & (weights > 0)
        & (cols >= 0)
        & (cols < n)
        & (cols != rows)
    )
    return rows[mask].astype(np.int32), cols[mask].astype(np.int32)


def degree_vectors(
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    n_nodes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    out_deg = np.bincount(edge_u, minlength=n_nodes).astype(np.int32)
    in_deg = np.bincount(edge_v, minlength=n_nodes).astype(np.int32)
    return out_deg, in_deg


def count_duplicates(edge_u: np.ndarray, edge_v: np.ndarray, n_nodes: int) -> int:
    if edge_u.size == 0:
        return 0
    enc = edge_u.astype(np.int64) * np.int64(n_nodes) + edge_v.astype(np.int64)
    return int(enc.size - np.unique(enc).size)


def rebuild_knn_from_edges(
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    n_nodes: int,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rebuild the same [N,K] KNN storage format.
    Since source nodes never change during target-swaps, each row retains
    exactly the same number of active edges as the Real Graph.
    """
    out_idx = np.tile(
        np.arange(n_nodes, dtype=np.uint16)[:, None],
        (1, k)
    )
    out_w = np.zeros((n_nodes, k), dtype=np.float32)

    counts = np.zeros(n_nodes, dtype=np.int32)
    for u, v in zip(edge_u, edge_v):
        pos = counts[u]
        if pos >= k:
            raise RuntimeError(
                f"Node {u} has out-degree {pos + 1} > K={k}. "
                "This should be impossible under degree-preserving rewiring."
            )
        out_idx[u, pos] = np.uint16(v)
        out_w[u, pos] = 1.0
        counts[u] += 1

    return out_idx, out_w


@njit(cache=True)
def _directed_double_edge_swap_numba(
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    n_nodes: int,
    target_swaps: int,
    max_attempts: int,
    seed: int,
):
    """
    Exact directed degree-preserving rewiring.

    Sources never change, so every node's out-degree is exactly preserved.
    Swapping targets b <-> d preserves every node's in-degree exactly.
    """
    np.random.seed(seed)

    m = edge_u.size
    if m < 2 or target_swaps <= 0:
        return edge_v, 0, 0, m

    # Current adjacency and original adjacency for overlap diagnostics.
    adj = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
    original_adj = np.zeros((n_nodes, n_nodes), dtype=np.uint8)

    for e in range(m):
        a = edge_u[e]
        b = edge_v[e]
        adj[a, b] = 1
        original_adj[a, b] = 1

    success = 0
    attempts = 0

    while success < target_swaps and attempts < max_attempts:
        attempts += 1

        e1 = np.random.randint(0, m)
        e2 = np.random.randint(0, m)

        if e1 == e2:
            continue

        a = edge_u[e1]
        b = edge_v[e1]
        c = edge_u[e2]
        d = edge_v[e2]

        # Same source or same target produces no useful target swap.
        if a == c or b == d:
            continue

        # Proposed edges: a->d and c->b. No self loops.
        if a == d or c == b:
            continue

        # Reject duplicates. The two original edges are still present here,
        # but because a!=c and b!=d the proposed edges are distinct from them.
        if adj[a, d] != 0 or adj[c, b] != 0:
            continue

        # Apply swap.
        adj[a, b] = 0
        adj[c, d] = 0
        adj[a, d] = 1
        adj[c, b] = 1

        edge_v[e1] = d
        edge_v[e2] = b

        success += 1

    # Count how many randomized edges still existed in the original graph.
    overlap = 0
    for e in range(m):
        if original_adj[edge_u[e], edge_v[e]] != 0:
            overlap += 1

    return edge_v, success, attempts, overlap


def randomize_one_day(
    nbr_idx_day: np.ndarray,
    nbr_w_day: np.ndarray,
    day_seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    n_nodes, k = nbr_idx_day.shape

    edge_u_orig, edge_v_orig = extract_active_directed_edges(
        nbr_idx_day, nbr_w_day
    )
    m = int(edge_u_orig.size)

    # Preserve original arrays for exact diagnostics.
    edge_u = edge_u_orig.copy()
    edge_v = edge_v_orig.copy()

    out_before, in_before = degree_vectors(edge_u_orig, edge_v_orig, n_nodes)

    target_swaps = int(round(SWAPS_PER_EDGE * m))
    max_attempts = int(
        max(target_swaps + 100, target_swaps * MAX_ATTEMPTS_PER_TARGET_SWAP)
    )

    edge_v, success, attempts, overlap = _directed_double_edge_swap_numba(
        edge_u=edge_u,
        edge_v=edge_v,
        n_nodes=n_nodes,
        target_swaps=target_swaps,
        max_attempts=max_attempts,
        seed=int(day_seed),
    )

    out_after, in_after = degree_vectors(edge_u, edge_v, n_nodes)

    out_exact = bool(np.array_equal(out_before, out_after))
    in_exact = bool(np.array_equal(in_before, in_after))

    self_loops = int(np.sum(edge_u == edge_v))
    duplicates = count_duplicates(edge_u, edge_v, n_nodes)

    idx_rand, w_rand = rebuild_knn_from_edges(
        edge_u=edge_u,
        edge_v=edge_v,
        n_nodes=n_nodes,
        k=k,
    )

    active_after = int(np.sum(w_rand > 0))
    overlap_ratio = float(overlap / m) if m > 0 else np.nan
    density = float(m / (n_nodes * max(n_nodes - 1, 1)))

    diagnostics = {
        "n_nodes": int(n_nodes),
        "k": int(k),
        "n_edges_real": m,
        "n_edges_random": active_after,
        "density": density,
        "target_successful_swaps": int(target_swaps),
        "successful_swaps": int(success),
        "attempted_swaps": int(attempts),
        "successful_swap_ratio_to_target": (
            float(success / target_swaps) if target_swaps > 0 else 1.0
        ),
        "edge_overlap_count": int(overlap),
        "edge_overlap_ratio": overlap_ratio,
        "out_degree_exact": out_exact,
        "in_degree_exact": in_exact,
        "self_loops": self_loops,
        "duplicate_edges": duplicates,
        "max_out_degree_real": int(out_before.max()) if n_nodes > 0 else 0,
        "max_out_degree_random": int(out_after.max()) if n_nodes > 0 else 0,
        "max_in_degree_real": int(in_before.max()) if n_nodes > 0 else 0,
        "max_in_degree_random": int(in_after.max()) if n_nodes > 0 else 0,
    }

    if REQUIRE_EXACT_OUT_DEGREE and not out_exact:
        raise RuntimeError("RandomGraph failed: out-degree sequence changed.")
    if REQUIRE_EXACT_IN_DEGREE and not in_exact:
        raise RuntimeError("RandomGraph failed: in-degree sequence changed.")
    if active_after != m:
        raise RuntimeError(
            f"RandomGraph failed: edge count changed ({m} -> {active_after})."
        )
    if REQUIRE_ZERO_SELF_LOOPS and self_loops != 0:
        raise RuntimeError(
            f"RandomGraph failed: {self_loops} self-loops were created."
        )
    if REQUIRE_ZERO_DUPLICATES and duplicates != 0:
        raise RuntimeError(
            f"RandomGraph failed: {duplicates} duplicate edges were created."
        )

    return idx_rand, w_rand, diagnostics


def day_specific_seed(graph_seed: int, day_index: int) -> int:
    # Deterministic, well-separated 31-bit seed.
    return int(
        (graph_seed * 1_000_003 + day_index * 91_769 + 12_345)
        % 2_147_483_647
    )


# ============================================================
# 2) SANITY MODE
# ============================================================

def run_sanity(
    dates: np.ndarray,
    nbr_idx: np.ndarray,
    nbr_w: np.ndarray,
):
    print("\n===== SANITY MODE =====")
    print("graph_seed =", SANITY_GRAPH_SEED)
    print("non-empty days requested =", SANITY_NONEMPTY_DAYS)

    rows = []
    checked = 0

    for t in range(len(dates)):
        active_edges = int(np.sum(nbr_w[t] > 0))
        if active_edges == 0:
            continue

        _, _, diag = randomize_one_day(
            nbr_idx_day=nbr_idx[t],
            nbr_w_day=nbr_w[t],
            day_seed=day_specific_seed(SANITY_GRAPH_SEED, t),
        )
        diag["t"] = int(t)
        diag["date"] = str(pd.Timestamp(dates[t]).date())
        rows.append(diag)
        checked += 1

        print(
            f"[SANITY] {diag['date']} | "
            f"E={diag['n_edges_real']} | "
            f"swaps={diag['successful_swaps']}/{diag['target_successful_swaps']} | "
            f"overlap={diag['edge_overlap_ratio']:.4f} | "
            f"out_exact={diag['out_degree_exact']} | "
            f"in_exact={diag['in_degree_exact']} | "
            f"self={diag['self_loops']} | dup={diag['duplicate_edges']}"
        )

        if checked >= SANITY_NONEMPTY_DAYS:
            break

    if not rows:
        raise RuntimeError("No non-empty graph days were found for sanity checking.")

    sanity_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sanity_path = OUT_DIR / "sanity_positive_degree_preserving_random_graph.csv"
    sanity_df.to_csv(sanity_path, index=False, encoding="utf-8-sig")

    max_overlap = float(sanity_df["edge_overlap_ratio"].max())
    all_out = bool(sanity_df["out_degree_exact"].all())
    all_in = bool(sanity_df["in_degree_exact"].all())
    all_self0 = bool((sanity_df["self_loops"] == 0).all())
    all_dup0 = bool((sanity_df["duplicate_edges"] == 0).all())

    print("\n[SANITY SUMMARY]")
    print("all out-degree exact =", all_out)
    print("all in-degree exact  =", all_in)
    print("all self-loops zero  =", all_self0)
    print("all duplicates zero  =", all_dup0)
    print("max edge overlap ratio =", max_overlap)
    print("saved ->", sanity_path)

    if max_overlap > WARN_OVERLAP_ABOVE:
        print(
            f"[WARNING] max edge-overlap ratio {max_overlap:.4f} "
            f"is above {WARN_OVERLAP_ABOVE:.2f}. "
            "Consider increasing SWAPS_PER_EDGE."
        )
    else:
        print(
            "[PASS] Randomization substantially changes edge identities "
            "while preserving degree and density exactly."
        )


# ============================================================
# 3) FULL MULTI-SEED GENERATION
# ============================================================

def generate_full_seed(
    graph_seed: int,
    dates: np.ndarray,
    nbr_idx_real: np.ndarray,
    nbr_w_real: np.ndarray,
    node_valid: np.ndarray,
    real_meta: Dict,
):
    seed_dir = OUT_DIR / f"seed{graph_seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    t_count, n_nodes, k = nbr_idx_real.shape

    # Same storage format as the Real Graph cache.
    nbr_idx_rand = np.tile(
        np.arange(n_nodes, dtype=np.uint16)[None, :, None],
        (t_count, 1, k)
    )
    nbr_w_rand = np.zeros((t_count, n_nodes, k), dtype=np.float32)

    rows = []
    t0 = time.perf_counter()

    for t in range(t_count):
        if np.sum(nbr_w_real[t] > 0) == 0:
            continue

        idx_t, w_t, diag = randomize_one_day(
            nbr_idx_day=nbr_idx_real[t],
            nbr_w_day=nbr_w_real[t],
            day_seed=day_specific_seed(graph_seed, t),
        )

        nbr_idx_rand[t] = idx_t
        nbr_w_rand[t] = w_t

        diag["t"] = int(t)
        diag["date"] = str(pd.Timestamp(dates[t]).date())
        diag["graph_seed"] = int(graph_seed)
        rows.append(diag)

        if (t + 1) % 50 == 0:
            print(
                f"[seed {graph_seed:02d}] day {t+1}/{t_count} | "
                f"date={diag['date']} | E={diag['n_edges_real']} | "
                f"overlap={diag['edge_overlap_ratio']:.4f}"
            )

    elapsed = float(time.perf_counter() - t0)
    diag_df = pd.DataFrame(rows)

    # Aggregate hard checks.
    if not diag_df.empty:
        if not bool(diag_df["out_degree_exact"].all()):
            raise RuntimeError(f"seed {graph_seed}: out-degree check failed.")
        if not bool(diag_df["in_degree_exact"].all()):
            raise RuntimeError(f"seed {graph_seed}: in-degree check failed.")
        if int(diag_df["self_loops"].sum()) != 0:
            raise RuntimeError(f"seed {graph_seed}: self-loop check failed.")
        if int(diag_df["duplicate_edges"].sum()) != 0:
            raise RuntimeError(f"seed {graph_seed}: duplicate-edge check failed.")

    random_meta = {
        "graph_type": "positive_degree_preserving_random_placebo",
        "base_graph_type": real_meta.get("graph_type", "unknown"),
        "graph_seed": int(graph_seed),
        "base_graph_variant": "positive_only",
        "base_positive_only": True,
        "rewiring": "directed_double_edge_swap",
        "preserves_exact_out_degree_per_node_per_day": True,
        "preserves_exact_in_degree_per_node_per_day": True,
        "preserves_exact_edge_count_per_day": True,
        "preserves_exact_density_per_day": True,
        "preserves_node_validity_mask": True,
        "self_loops_allowed": False,
        "duplicate_edges_allowed": False,
        "swaps_per_edge": float(SWAPS_PER_EDGE),
        "max_attempts_per_target_swap": int(MAX_ATTEMPTS_PER_TARGET_SWAP),
        "mean_edge_overlap_ratio": (
            float(diag_df["edge_overlap_ratio"].mean())
            if not diag_df.empty else np.nan
        ),
        "max_edge_overlap_ratio": (
            float(diag_df["edge_overlap_ratio"].max())
            if not diag_df.empty else np.nan
        ),
        "generation_seconds": elapsed,
        "note": (
            "Random graph is generated from the Positive-only Graph-2 computational "
            "KNN adjacency. Real cross-stock edge identities are rewired while "
            "daily in/out degrees and density are preserved exactly."
        ),
    }

    cache_path = seed_dir / (
        f"positive_random_degree_preserving_seed{graph_seed:02d}_"
        f"T{t_count}_N{n_nodes}_K{k}.npz"
    )
    np.savez_compressed(
        cache_path,
        dates=dates.astype("datetime64[ns]"),
        nbr_idx=nbr_idx_rand.astype(np.uint16),
        nbr_w=nbr_w_rand.astype(np.float32),
        node_valid=node_valid.astype(np.uint8),
        meta=json.dumps(random_meta, ensure_ascii=False),
    )

    diag_path = seed_dir / f"positive_diagnostics_seed{graph_seed:02d}.csv"
    diag_df.to_csv(diag_path, index=False, encoding="utf-8-sig")

    meta_path = seed_dir / f"positive_meta_seed{graph_seed:02d}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(random_meta, f, ensure_ascii=False, indent=2)

    print(
        f"\n[DONE seed {graph_seed:02d}] "
        f"time={elapsed:.1f}s | "
        f"mean_overlap={random_meta['mean_edge_overlap_ratio']:.4f} | "
        f"max_overlap={random_meta['max_edge_overlap_ratio']:.4f}"
    )
    print("cache ->", cache_path)
    print("diagnostics ->", diag_path)

    return random_meta


# ============================================================
# 4) MAIN
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    real_cache = resolve_real_graph_cache()
    print("[INFO] Positive-only Real Graph-2 cache =", real_cache)

    pack = np.load(real_cache, allow_pickle=True)
    dates = pack["dates"].astype("datetime64[ns]")
    nbr_idx = pack["nbr_idx"]
    nbr_w = pack["nbr_w"].astype(np.float32)
    node_valid = pack["node_valid"].astype(np.uint8)

    try:
        real_meta = json.loads(str(pack["meta"].item()))
    except Exception:
        real_meta = {}

    # Hard check: this placebo MUST be generated from the Positive-only graph.
    graph_type = str(real_meta.get("graph_type", ""))
    positive_only = bool(real_meta.get("positive_only", False))
    abs_corr = bool(real_meta.get("abs_corr", False))

    if ("positive" not in graph_type.lower()) and (not positive_only):
        raise RuntimeError(
            "Base graph is not identified as Positive-only. "
            f"graph_type={graph_type!r}, positive_only={positive_only}"
        )
    if abs_corr:
        raise RuntimeError(
            "Base graph metadata says abs_corr=True. "
            "Do not use the old absolute-correlation graph for this placebo."
        )

    print("[INFO] verified base graph: Positive-only")
    print("[INFO] base graph_type =", graph_type)
    print("[INFO] base positive_only =", positive_only, "| abs_corr =", abs_corr)

    print(
        f"[INFO] shape: T={nbr_idx.shape[0]} "
        f"N={nbr_idx.shape[1]} K={nbr_idx.shape[2]}"
    )
    print("[INFO] SWAPS_PER_EDGE =", SWAPS_PER_EDGE)

    if SANITY_ONLY:
        run_sanity(dates, nbr_idx, nbr_w)
        print(
            "\nSanity finished. If all exact-degree/self-loop/duplicate checks pass "
            "and edge overlap is sufficiently low, set SANITY_ONLY=False "
            "and rerun to generate all GRAPH_SEEDS."
        )
        return

    summaries: List[Dict] = []
    for graph_seed in GRAPH_SEEDS:
        summaries.append(
            generate_full_seed(
                graph_seed=graph_seed,
                dates=dates,
                nbr_idx_real=nbr_idx,
                nbr_w_real=nbr_w,
                node_valid=node_valid,
                real_meta=real_meta,
            )
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = OUT_DIR / "positive_random_graph_generation_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n===== ALL RANDOM GRAPH SEEDS COMPLETE =====")
    print(summary_df[
        [
            "graph_seed",
            "mean_edge_overlap_ratio",
            "max_edge_overlap_ratio",
            "generation_seconds",
        ]
    ].to_string(index=False))
    print("summary ->", summary_path)


if __name__ == "__main__":
    main()
