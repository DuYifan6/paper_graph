
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from project_config import OUTPUT_ROOT, common_cache_paths


SCRIPT_DIR = Path(__file__).resolve().parent
GRAPH2_SCRIPT = SCRIPT_DIR / "03_graph2.py"
DEFAULT_GRAPH2_CACHE = (
    OUTPUT_ROOT
    / "graph_assets"
    / "graph2_positive_only"
    / "graphs"
    / "roll20_binary_positive_mkt1_topk40_shift0_tau0.1_pow1.0_full1.npz"
)
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "graph2_ablation"

VARIANTS = ("full", "no_laplacian", "no_gate", "identity", "random")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final-protocol Graph-2 structural ablations."
    )
    parser.add_argument(
        "--variant",
        choices=("all", *VARIANTS),
        default="all",
        help="Ablation to run; 'all' runs every listed variant.",
    )
    parser.add_argument(
        "--graph2-cache",
        type=Path,
        default=DEFAULT_GRAPH2_CACHE,
        help="Final empirical positive-only binary Top-40 Graph-2 cache.",
    )
    parser.add_argument(
        "--random-graph-cache",
        type=Path,
        default=None,
        help="Prebuilt degree-preserving RandomGraph cache; required for random/all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for variant-specific checkpoints and results.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run one epoch for pipeline checking; do not report as final results.",
    )
    parser.add_argument(
        "--allow-legacy-graph-cache",
        action="store_true",
        help=(
            "Allow a graph cache without an embedded stock-order array. Use only "
            "after independently verifying that its node order matches the common cache."
        ),
    )
    return parser.parse_args(argv)


def load_graph2_module():
    if not GRAPH2_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Missing companion script: {GRAPH2_SCRIPT}. Keep this file beside 03_graph2.py."
        )
    spec = importlib.util.spec_from_file_location("final_graph2", GRAPH2_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {GRAPH2_SCRIPT}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NoGateGraphMix(nn.Module):
    """GraphMix with the gate fixed to one and all parameters still registered."""

    def __init__(self, hidden: int, gate_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.lin_gate = nn.Linear(hidden * 2, gate_hidden)
        self.lin_gate2 = nn.Linear(gate_hidden, hidden)
        self.lin_msg = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Z, nbr_idx=None, nbr_w=None, node_valid_mask=None):
        if nbr_idx is None or nbr_w is None:
            return Z
        n_nodes, hidden = Z.shape
        n_neighbors = nbr_idx.shape[1]
        z_send = Z if node_valid_mask is None else Z * node_valid_mask.view(n_nodes, 1)
        weights = torch.nan_to_num(
            nbr_w, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        neighbors = z_send.index_select(0, nbr_idx.reshape(-1)).view(
            n_nodes, n_neighbors, hidden
        )
        numerator = (weights.unsqueeze(-1) * neighbors).sum(dim=1)
        aggregate = numerator / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        combined = torch.cat([Z, aggregate], dim=1)
        delta = self.dropout(F.relu(self.lin_msg(combined)))
        return Z + delta


class IdentityGraphMix(nn.Module):
    """Parameter-matched GraphMix that is deliberately bypassed."""

    def __init__(self, hidden: int, gate_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.lin_gate = nn.Linear(hidden * 2, gate_hidden)
        self.lin_gate2 = nn.Linear(gate_hidden, hidden)
        self.lin_msg = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Z, nbr_idx=None, nbr_w=None, node_valid_mask=None):
        return Z


def decode_graph_meta(pack: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    if "meta" not in pack.files:
        return {}
    raw = pack["meta"]
    try:
        value = raw.item() if getattr(raw, "shape", None) == () else raw.tolist()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            return json.loads(value)
        return dict(value) if isinstance(value, dict) else {}
    except Exception:
        return {}


def load_graph_cache(
    path: Path,
    dates: Sequence[pd.Timestamp],
    codes: Sequence[str],
    *,
    cache_role: str,
    allow_legacy: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{cache_role} cache not found: {path}")

    with np.load(path, allow_pickle=True) as pack:
        required = {"dates", "nbr_idx", "nbr_w"}
        missing = sorted(required - set(pack.files))
        if missing:
            raise KeyError(f"{cache_role} cache is missing arrays: {missing}")

        graph_dates = pack["dates"].astype("datetime64[ns]")
        nbr_idx = np.asarray(pack["nbr_idx"])
        nbr_w = np.asarray(pack["nbr_w"], dtype=np.float32)
        graph_meta = decode_graph_meta(pack)
        cached_codes = (
            pack["stocks"].astype(str).tolist() if "stocks" in pack.files else None
        )

    expected_dates = pd.to_datetime(pd.Series(dates)).to_numpy(dtype="datetime64[ns]")
    expected_shape = (len(dates), len(codes), min(40, max(1, len(codes) - 1)))

    if not np.array_equal(graph_dates, expected_dates):
        raise RuntimeError(f"{cache_role} dates do not match the common cache.")
    if nbr_idx.shape != expected_shape or nbr_w.shape != expected_shape:
        raise RuntimeError(
            f"{cache_role} shape mismatch: expected {expected_shape}, "
            f"got idx={nbr_idx.shape}, w={nbr_w.shape}."
        )
    if cached_codes is None:
        if not allow_legacy:
            raise RuntimeError(
                f"{cache_role} does not store stock order. Rebuild it with the final "
                "code, or pass --allow-legacy-graph-cache only after manual verification."
            )
        print(f"[WARN] {cache_role}: accepting an unverified legacy stock order.")
    elif cached_codes != list(codes):
        raise RuntimeError(f"{cache_role} stock order does not match the common cache.")

    if np.any(nbr_idx < 0) or np.any(nbr_idx >= len(codes)):
        raise RuntimeError(f"{cache_role} contains out-of-range neighbor indices.")
    if not np.isfinite(nbr_w).all() or np.any(nbr_w < 0):
        raise RuntimeError(f"{cache_role} contains invalid edge weights.")
    if int(graph_meta.get("topk", expected_shape[2])) != expected_shape[2]:
        raise RuntimeError(f"{cache_role} metadata is not Top-s=40.")

    if cache_role == "Graph-2":
        checks = {
            "roll_window_days": 20,
            "positive_only": True,
            "edge_tau": 0.10,
        }
        for key, expected in checks.items():
            if graph_meta.get(key) != expected:
                raise RuntimeError(
                    f"Graph-2 metadata mismatch for {key}: expected {expected!r}, "
                    f"got {graph_meta.get(key)!r}."
                )
        if not str(graph_meta.get("graph_type", "")).endswith(
            "binary_from_5m_logret"
        ):
            raise RuntimeError("The empirical cache is not the final binary Graph-2.")

    return graph_dates, nbr_idx, nbr_w, graph_meta


def variant_settings(base, variant: str):
    if variant == "identity":
        return IdentityGraphMix, 0.0, "identity/self-only; GraphMix bypassed"
    if variant == "no_gate":
        return NoGateGraphMix, 1e-3, "empirical Graph-2; gate fixed to one"
    if variant == "no_laplacian":
        return base.GraphMixKNN, 0.0, "empirical Graph-2; Laplacian removed"
    if variant == "random":
        return base.GraphMixKNN, 1e-3, "degree-preserving RandomGraph"
    if variant == "full":
        return base.GraphMixKNN, 1e-3, "final empirical Graph-2"
    raise ValueError(f"Unknown variant: {variant}")


def graphmix_gradient_audit(model: nn.Module) -> Dict[str, int]:
    registered = sum(p.numel() for p in model.graph.parameters() if p.requires_grad)
    with_gradient = sum(
        p.numel()
        for p in model.graph.parameters()
        if p.requires_grad and p.grad is not None
    )
    nonzero_gradient = sum(
        p.numel()
        for p in model.graph.parameters()
        if p.requires_grad
        and p.grad is not None
        and bool(torch.any(p.grad.detach() != 0).item())
    )
    return {
        "graphmix_registered_params": int(registered),
        "graphmix_params_with_gradient": int(with_gradient),
        "graphmix_params_with_nonzero_gradient": int(nonzero_gradient),
    }


def save_evaluation(result: Dict[str, Any], directory: Path, split: str) -> None:
    result["detail_df"].to_csv(
        directory / f"pred_{split}.csv", index=False, encoding="utf-8-sig"
    )
    result["daily_df"].to_csv(
        directory / f"daily_{split}.csv", index=False, encoding="utf-8-sig"
    )


def run_variant(
    base,
    variant: str,
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    dates: Sequence[pd.Timestamp],
    codes: Sequence[str],
    split: Tuple[int, int, int, int, int, int],
    empirical_graph: Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]],
    random_graph: Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]] | None,
    output_root: Path,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    train_start, fit_end, val_start, val_end, test_start, test_end = split
    graphmix_class, laplacian_lambda, description = variant_settings(base, variant)
    chosen_graph = random_graph if variant == "random" else empirical_graph
    if chosen_graph is None:
        raise RuntimeError("RandomGraph cache is required for the random variant.")
    graph_dates, nbr_idx, nbr_w, graph_meta = chosen_graph

    original_graphmix_class = base.GraphMixKNN
    original_laplacian = base.LAPLACIAN_LAM
    try:
        base.GraphMixKNN = graphmix_class
        base.LAPLACIAN_LAM = laplacian_lambda
        print(f"\n===== {variant} =====")
        print(f"[CONFIG] {description}; lambda_lap={laplacian_lambda:g}")
        model, history, split_info = base.train_model_on_range(
            X,
            y,
            mask,
            dates,
            codes,
            graph_dates,
            nbr_idx,
            nbr_w,
            in_dim=X.shape[2],
            device=device,
            train_start_idx=train_start,
            fit_end_idx=fit_end,
            val_start_idx=val_start,
            val_end_idx=val_end,
            seed=seed,
        )
    finally:
        base.GraphMixKNN = original_graphmix_class
        base.LAPLACIAN_LAM = original_laplacian

    loss_curve = history["val_mse"].astype(float).tolist()
    val_result = base.evaluate_model_on_range(
        model,
        X,
        y,
        mask,
        dates,
        codes,
        graph_dates,
        nbr_idx,
        nbr_w,
        device=device,
        eval_start_idx=val_start,
        eval_end_idx=val_end,
        loss_curve=loss_curve,
    )
    test_result = base.evaluate_model_on_range(
        model,
        X,
        y,
        mask,
        dates,
        codes,
        graph_dates,
        nbr_idx,
        nbr_w,
        device=device,
        eval_start_idx=test_start,
        eval_end_idx=test_end,
        loss_curve=loss_curve,
    )
    if val_result is None or test_result is None:
        raise RuntimeError(f"{variant}: evaluation produced no valid observations.")

    gradient_audit = graphmix_gradient_audit(model)
    if variant == "identity" and gradient_audit["graphmix_params_with_gradient"] != 0:
        raise RuntimeError("IdentityGraph unexpectedly produced GraphMix gradients.")

    variant_dir = output_root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(
        variant_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    save_evaluation(val_result, variant_dir, "validation")
    save_evaluation(test_result, variant_dir, "test")
    torch.save(model.state_dict(), variant_dir / "best_model.pt")

    config = {
        "variant": variant,
        "description": description,
        "training_loss": "MSE + lambda_lap * normalized Laplacian",
        "ranking_loss": False,
        "laplacian_lambda": laplacian_lambda,
        "selection_criterion": "minimum pooled validation MSE",
        "lookback": base.LOOKBACK,
        "epochs": 1 if base.SANITY_ONLY else base.EPOCHS,
        "patience": base.EARLY_STOP_PATIENCE,
        "optimizer": "Adam",
        "learning_rate": base.LR,
        "weight_decay": base.WEIGHT_DECAY,
        "seed": seed,
        "graph_meta": graph_meta,
        **split_info,
        **gradient_audit,
    }
    with (variant_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)

    summary = {
        "variant": variant,
        "best_epoch": split_info["best_epoch"],
        "validation_mse": val_result["mse"],
        "validation_r2": val_result["r2"],
        "test_mse": test_result["mse"],
        "test_r2": test_result["r2"],
        "n_test": test_result["n_obs"],
        "laplacian_lambda": laplacian_lambda,
        **gradient_audit,
    }
    pd.DataFrame([summary]).to_csv(
        variant_dir / "summary.csv", index=False, encoding="utf-8-sig"
    )
    print(pd.DataFrame([summary]).to_string(index=False))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    if "random" in variants and args.random_graph_cache is None:
        raise ValueError(
            "--random-graph-cache is required for --variant random/all. "
            "It must be the prebuilt degree-preserving Top-40 cache."
        )

    base = load_graph2_module()
    base.SEED = args.seed
    base.SANITY_ONLY = bool(args.sanity)
    base.USE_RANK_LOSS = False
    base.RANK_LOSS_LAM = 0.0
    base.USE_NORMALIZED_LAPLACIAN = True
    base.USE_SYM_NORM_IN_GRAPHMIX = True

    cache_path, meta_path = common_cache_paths()
    if not cache_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"Common cache is missing: {cache_path} / {meta_path}. "
            "Run 00_build_common_cache.py first."
        )
    with np.load(cache_path) as pack:
        X = pack["X_np"].astype(np.float32)
        y = pack["y_np"].astype(np.float32)
        mask = pack["mask_np"].astype(bool)
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)

    dates = [pd.Timestamp(value).normalize() for value in meta["all_dates"]]
    codes = list(meta["ts_codes_order"])
    if X.shape[:2] != y.shape or y.shape != mask.shape:
        raise RuntimeError(
            f"Common-cache shape mismatch: X={X.shape}, y={y.shape}, mask={mask.shape}."
        )
    if X.shape[0] != len(dates) or X.shape[1] != len(codes):
        raise RuntimeError("Common-cache metadata order does not match tensor shape.")

    split = base.resolve_split(dates, meta)
    train_start, fit_end, val_start, val_end, test_start, test_end = split
    print(f"[DEVICE] {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"[FIT] {dates[max(train_start, base.LOOKBACK - 1)].date()} .. {dates[fit_end].date()}")
    print(f"[VALIDATION] {dates[val_start].date()} .. {dates[val_end].date()}")
    print(f"[TEST] {dates[test_start].date()} .. {dates[test_end].date()}")

    empirical_graph = load_graph_cache(
        args.graph2_cache,
        dates,
        codes,
        cache_role="Graph-2",
        allow_legacy=args.allow_legacy_graph_cache,
    )
    random_graph = None
    if "random" in variants:
        random_graph = load_graph_cache(
            args.random_graph_cache,
            dates,
            codes,
            cache_role="RandomGraph",
            allow_legacy=args.allow_legacy_graph_cache,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = []
    for variant in variants:
        summaries.append(
            run_variant(
                base,
                variant,
                X,
                y,
                mask,
                dates,
                codes,
                split,
                empirical_graph,
                random_graph,
                args.output_dir,
                device,
                args.seed,
            )
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(
        args.output_dir / "summary_all_variants.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\nFINAL GRAPH-2 ABLATION SUMMARY")
    print(summary_df.to_string(index=False))
    if args.sanity:
        print("[WARNING] --sanity results are pipeline checks, not reportable estimates.")


if __name__ == "__main__":
    main()
