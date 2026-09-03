# -*- coding: utf-8 -*-
"""
GCN-LSTM

"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import json
import pickle
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import torch
import torch.nn as nn

ROOT = Path(r"D:\PythonProject\LASSO_FINAL")
FAIR_ROOT = ROOT / "AE-3" / "benchmark_fairness"
CACHE_NPZ = ROOT / "common_preprocess_cache" / "common_tensor_cache.npz"
META_PKL = ROOT / "common_preprocess_cache" / "common_tensor_meta.pkl"
# This must be the exact cache read by the final Graph-2 MSE-fair script.
# Do not use filename globbing or a fallback graph: those can silently change
# W, s, tau, sign treatment, or stock/date ordering.
GRAPH_CACHE = (
    ROOT
    / "graph_大修"
    / "graph2_positive_only"
    / "graphs"
    / "roll20_binary_positive_mkt1_topk40_shift0_tau0.1_pow1.0_ddof1_full1.npz"
)

OUT_DIR = FAIR_ROOT / "gcn_lstm"
RESULT_DIR = OUT_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 20
EXPECTED_FEATURES = 9
GRAPH_ROLL_WINDOW = 20
GRAPH_TOPK = 40
GRAPH_TAU = 0.10
GRAPH_MARKET_REMOVE = True
GRAPH_CORR_STD_DDOF = 1
GCN_HIDDEN = 64
LSTM_HIDDEN = 128
DROPOUT = 0.10
LR = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
PATIENCE = 4
GRAD_CLIP = 1.0
MIN_VALID_NODES_PER_DAY = 50
SEED = 0

VAL_START_DATE = "2024-06-06"
FIT_END_DATE = "2024-06-05"
VAL_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-02"
TEST_END_DATE = "2025-11-17"
EXPECTED_TEST_DAYS = 211

def resolve_split(dates, meta):
    """Use the original common cache but impose one common model-selection split."""
    train_start = int(meta["train_start_idx"])
    train_end = int(meta["train_end_idx"])
    test_start = int(meta["test_start_idx"])
    test_end = int(meta["test_end_idx"])

    date_arr = pd.DatetimeIndex(dates)
    candidates = np.where(date_arr >= pd.Timestamp(VAL_START_DATE))[0]
    candidates = candidates[(candidates >= train_start) & (candidates <= train_end)]
    if len(candidates) == 0:
        raise ValueError(f"Cannot locate validation start {VAL_START_DATE} in pre-2025 dates.")
    val_start = int(candidates[0])
    fit_end = val_start - 1

    first_fit_target = max(train_start, LOOKBACK - 1)
    if fit_end < first_fit_target:
        raise ValueError("Fitting period is too short after applying LOOKBACK.")

    return train_start, fit_end, val_start, train_end, test_start, test_end


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class NeighborGCNLayer(nn.Module):
    """h_i = W_self x_i + W_nei sum_j normalized(w_ij) x_j"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.nei_lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, nbr_idx: torch.Tensor, nbr_w: torch.Tensor) -> torch.Tensor:
        neigh = x[nbr_idx]
        w = torch.nan_to_num(nbr_w.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        agg = (neigh * w.unsqueeze(-1)).sum(dim=1)
        h = self.self_lin(x) + self.nei_lin(agg)
        return self.dropout(torch.relu(self.norm(h)))


class GCNLSTM(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.gcn1 = NeighborGCNLayer(in_dim, GCN_HIDDEN, DROPOUT)
        self.gcn2 = NeighborGCNLayer(GCN_HIDDEN, GCN_HIDDEN, DROPOUT)
        self.lstm = nn.LSTM(GCN_HIDDEN, LSTM_HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(LSTM_HIDDEN, LSTM_HIDDEN // 2),
            nn.ReLU(),
            nn.Linear(LSTM_HIDDEN // 2, 1),
        )

    def forward(self, x_seq: torch.Tensor, nbr_idx_seq: torch.Tensor, nbr_w_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [L,N,F]; graphs: [L,N,K]
        hs = []
        for u in range(x_seq.shape[0]):
            h = self.gcn1(x_seq[u], nbr_idx_seq[u], nbr_w_seq[u])
            h = self.gcn2(h, nbr_idx_seq[u], nbr_w_seq[u])
            hs.append(h)
        z = torch.stack(hs, dim=1)  # [N,L,H]
        out, _ = self.lstm(z)
        return self.head(out[:, -1, :]).squeeze(-1)


class GraphStore:
    """Validated view of the exact positive-only binary Graph-2 cache."""

    def __init__(self, path: Path, dates: List[pd.Timestamp], codes: List[str]):
        with np.load(path, allow_pickle=True) as z:
            required_arrays = {"dates", "nbr_idx", "nbr_w", "node_valid", "meta"}
            missing = required_arrays.difference(z.files)
            if missing:
                raise RuntimeError(
                    f"Graph-2 cache is missing required arrays: {sorted(missing)}."
                )

            graph_dates = z["dates"].astype("datetime64[ns]")
            self.nbr_idx = z["nbr_idx"]
            self.nbr_w = z["nbr_w"].astype(np.float32)
            self.node_valid = z["node_valid"].astype(bool)
            try:
                graph_meta = json.loads(str(z["meta"].item()))
            except Exception as exc:
                raise RuntimeError("Graph-2 cache metadata is not valid JSON.") from exc

            expected_dates = pd.DatetimeIndex(dates).to_numpy(dtype="datetime64[ns]")
            if not np.array_equal(graph_dates, expected_dates):
                raise RuntimeError(
                    "Graph-2 cache dates are not exactly aligned with the common tensor."
                )

            if "stocks" in z.files:
                graph_codes = z["stocks"].astype(str).tolist()
                if graph_codes != list(codes):
                    raise RuntimeError(
                        "Graph-2 cache stock order does not match the common tensor."
                    )

        expected_topk = min(GRAPH_TOPK, max(1, len(codes) - 1))
        expected_shape = (len(dates), len(codes), expected_topk)
        if self.nbr_idx.shape != expected_shape or self.nbr_w.shape != expected_shape:
            raise RuntimeError(
                f"Graph-2 cache shape mismatch: idx={self.nbr_idx.shape}, "
                f"weights={self.nbr_w.shape}, expected={expected_shape}."
            )
        if self.node_valid.shape != (len(dates), len(codes)):
            raise RuntimeError(
                f"Graph-2 node_valid shape mismatch: got {self.node_valid.shape}."
            )

        required_meta = {
            "roll_window_days": GRAPH_ROLL_WINDOW,
            "topk": expected_topk,
            "positive_only": True,
            "market_remove": GRAPH_MARKET_REMOVE,
            "edge_tau": GRAPH_TAU,
            "corr_std_ddof": GRAPH_CORR_STD_DDOF,
        }
        for key, expected in required_meta.items():
            actual = graph_meta.get(key)
            if isinstance(expected, float):
                matches = actual is not None and np.isclose(
                    float(actual), expected, rtol=0.0, atol=1e-12
                )
            else:
                matches = actual == expected
            if not matches:
                raise RuntimeError(
                    f"Graph-2 cache metadata mismatch for {key}: "
                    f"expected {expected!r}, got {actual!r}."
                )

        graph_type = str(graph_meta.get("graph_type", ""))
        if not graph_type.endswith("binary_from_5m_logret"):
            raise RuntimeError(
                f"GCN-LSTM requires the binary Graph-2 cache; got {graph_type!r}."
            )
        self._validate_binary_edges()

    def _validate_binary_edges(self) -> None:
        """Check weights and indices in daily chunks to avoid a large temp array."""
        n_nodes = self.nbr_idx.shape[1]
        rows = np.arange(n_nodes, dtype=np.int64)[:, None]
        for t in range(self.nbr_idx.shape[0]):
            idx = np.asarray(self.nbr_idx[t], dtype=np.int64)
            w = np.asarray(self.nbr_w[t], dtype=np.float32)
            if not np.isfinite(w).all() or not np.logical_or(w == 0.0, w == 1.0).all():
                raise RuntimeError(
                    f"Graph-2 cache must be binary; invalid weight found at day index {t}."
                )
            active = w > 0.0
            if active.any():
                if (idx[active] < 0).any() or (idx[active] >= n_nodes).any():
                    raise RuntimeError(
                        f"Graph-2 cache contains an out-of-range neighbor at day index {t}."
                    )
                if (idx[active] == np.broadcast_to(rows, idx.shape)[active]).any():
                    raise RuntimeError(
                        f"Graph-2 cache contains an active self-loop at day index {t}."
                    )

    def seq(self, t: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Correct alignment: same date range as X[t-L+1:t+1].
        ni = np.asarray(self.nbr_idx[t - LOOKBACK + 1:t + 1], dtype=np.int64)
        nw = np.asarray(self.nbr_w[t - LOOKBACK + 1:t + 1], dtype=np.float32)
        return torch.as_tensor(ni, dtype=torch.long, device=device), torch.as_tensor(nw, dtype=torch.float32, device=device)


def prepare_x_seq(X: np.ndarray, t: int, device: torch.device) -> torch.Tensor:
    # X[t] is already t-1 information; include X[t] as the latest lagged predictor row.
    x = np.asarray(X[t - LOOKBACK + 1:t + 1], dtype=np.float32)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def validate_common_inputs(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    dates: List[pd.Timestamp],
    codes: List[str],
) -> None:
    if X.ndim != 3 or y.ndim != 2 or mask.ndim != 2:
        raise RuntimeError(
            f"Unexpected tensor ranks: X={X.shape}, y={y.shape}, mask={mask.shape}."
        )
    if X.shape[:2] != y.shape or y.shape != mask.shape:
        raise RuntimeError(
            f"Common-cache shape mismatch: X={X.shape}, y={y.shape}, mask={mask.shape}."
        )
    if X.shape[2] != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Expected the fixed {EXPECTED_FEATURES}-predictor space, got {X.shape[2]}."
        )
    if X.shape[:2] != (len(dates), len(codes)):
        raise RuntimeError(
            f"Tensor/metadata mismatch: X={X.shape}, dates={len(dates)}, stocks={len(codes)}."
        )
    date_index = pd.DatetimeIndex(dates)
    if date_index.has_duplicates or not date_index.is_monotonic_increasing:
        raise RuntimeError("Common-cache dates must be unique and strictly increasing.")


def validate_paper_split(
    dates: List[pd.Timestamp],
    fit_end: int,
    val_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    test_days: np.ndarray,
) -> None:
    expected = {
        "fit_end": (fit_end, FIT_END_DATE),
        "val_start": (val_start, VAL_START_DATE),
        "val_end": (train_end, VAL_END_DATE),
        "test_start": (test_start, TEST_START_DATE),
        "test_end": (test_end, TEST_END_DATE),
    }
    for label, (idx, expected_date) in expected.items():
        actual = str(pd.Timestamp(dates[idx]).date())
        if actual != expected_date:
            raise RuntimeError(
                f"{label} mismatch: expected {expected_date}, got {actual}."
            )
    if len(test_days) != EXPECTED_TEST_DAYS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_DAYS} OOS target days, got {len(test_days)}."
        )


def evaluate_mse(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    graph: GraphStore,
    days: np.ndarray,
    device: torch.device,
    save_predictions: bool = False,
    dates: List[pd.Timestamp] | None = None,
    codes: List[str] | None = None,
):
    model.eval()
    sse, n = 0.0, 0
    rows = []
    with torch.no_grad():
        for t in map(int, days):
            m_np = mask[t]
            n_t = int(m_np.sum())
            if n_t < MIN_VALID_NODES_PER_DAY:
                continue
            x = prepare_x_seq(X, t, device)
            ni, nw = graph.seq(t, device)
            yy = torch.as_tensor(np.asarray(y[t], dtype=np.float32), device=device)
            m = torch.as_tensor(m_np, dtype=torch.bool, device=device)
            pred = model(x, ni, nw)
            err = pred[m] - yy[m]
            sse += float(torch.sum(err * err).item())
            n += n_t

            if save_predictions:
                idx = np.flatnonzero(m_np)
                rows.append(pd.DataFrame({
                    "date": str(pd.Timestamp(dates[t]).date()),
                    "stock": np.asarray(codes, dtype=object)[idx].astype(str),
                    "y_true": np.asarray(y[t, idx], dtype=np.float32),
                    "y_pred": pred.detach().cpu().numpy()[idx].astype(np.float32),
                    "model": "GCN_LSTM_MSEFair",
                }))
    mse = float(sse / n) if n else float("inf")
    df = pd.concat(rows, ignore_index=True) if rows else None
    return mse, df


def main():
    if not GRAPH_CACHE.exists():
        raise FileNotFoundError(
            f"Final Graph-2 cache not found: {GRAPH_CACHE}\n"
            "Run the final Graph-2 MSE-fair script first; no fallback graph is allowed."
        )
    print(f"[INFO] Graph cache = {GRAPH_CACHE}")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pack = np.load(CACHE_NPZ)
    X = pack["X_np"].astype(np.float32)
    y = pack["y_np"].astype(np.float32)
    mask = pack["mask_np"].astype(bool)
    with open(META_PKL, "rb") as f:
        meta = pickle.load(f)

    dates = [pd.Timestamp(d).normalize() for d in meta["all_dates"]]
    codes = list(meta["ts_codes_order"])
    validate_common_inputs(X, y, mask, dates, codes)
    train_start, fit_end, val_start, train_end, test_start, test_end = resolve_split(dates, meta)
    print(f"[SPLIT] fit targets: {dates[max(train_start, LOOKBACK - 1)].date()} .. {dates[fit_end].date()}")
    print(f"[SPLIT] validation: {dates[val_start].date()} .. {dates[train_end].date()}")
    print(f"[SPLIT] test: {dates[test_start].date()} .. {dates[test_end].date()}")

    fit_days = np.arange(max(LOOKBACK - 1, train_start), fit_end + 1, dtype=np.int32)
    val_days = np.arange(max(LOOKBACK - 1, val_start), train_end + 1, dtype=np.int32)
    test_days = np.arange(max(LOOKBACK - 1, test_start), test_end + 1, dtype=np.int32)
    validate_paper_split(
        dates, fit_end, val_start, train_end, test_start, test_end, test_days
    )

    graph = GraphStore(GRAPH_CACHE, dates, codes)

    model = GCNLSTM(X.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0
    history = []

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        shuffled = fit_days.copy()
        np.random.default_rng(SEED + ep).shuffle(shuffled)
        train_sse, train_n, used_days = 0.0, 0, 0

        for t in map(int, shuffled):
            m_np = mask[t]
            n_t = int(m_np.sum())
            if n_t < MIN_VALID_NODES_PER_DAY:
                continue
            x = prepare_x_seq(X, t, device)
            ni, nw = graph.seq(t, device)
            yy = torch.as_tensor(np.asarray(y[t], dtype=np.float32), device=device)
            m = torch.as_tensor(m_np, dtype=torch.bool, device=device)

            opt.zero_grad(set_to_none=True)
            pred = model(x, ni, nw)
            loss = torch.mean((pred[m] - yy[m]) ** 2)
            loss.backward()
            if GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            train_sse += float(loss.item()) * n_t
            train_n += n_t
            used_days += 1

        train_mse = train_sse / max(train_n, 1)
        val_mse, _ = evaluate_mse(model, X, y, mask, graph, val_days, device)
        history.append({
            "epoch": ep,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "used_days": used_days,
            "elapsed_sec": time.time() - t0,
        })
        print(f"epoch {ep:02d}/{EPOCHS} train_mse={train_mse:.8f} val_mse={val_mse:.8f} used_days={used_days}")

        if val_mse < best_val - 1e-8:
            best_val = val_mse
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[EARLY STOP] epoch={ep}, best_epoch={best_epoch}, best_val_mse={best_val:.8f}")
                break

    if best_state is None:
        raise RuntimeError("No checkpoint selected.")
    model.load_state_dict(best_state)
    torch.save(best_state, RESULT_DIR / "best_gcn_lstm_msefair.pt")
    pd.DataFrame(history).to_csv(RESULT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

    val_mse, val_df = evaluate_mse(model, X, y, mask, graph, val_days, device, True, dates, codes)
    test_mse, test_df = evaluate_mse(model, X, y, mask, graph, test_days, device, True, dates, codes)
    val_df.to_csv(RESULT_DIR / "pred_gcn_lstm_validation.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(RESULT_DIR / "pred_gcn_lstm_test.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        "model": "GCN_LSTM_MSEFair",
        "selection_criterion": "minimum validation MSE",
        "training_loss": "MSE",
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "validation_mse_recomputed": val_mse,
        "validation_r2": float(r2_score(val_df["y_true"], val_df["y_pred"])),
        "test_mse": test_mse,
        "test_r2": float(r2_score(test_df["y_true"], test_df["y_pred"])),
        "n_test": len(test_df),
        "gcn_hidden": GCN_HIDDEN,
        "lstm_hidden": LSTM_HIDDEN,
        "lookback": LOOKBACK,
        "max_epochs": EPOCHS,
        "patience": PATIENCE,
        "seed": SEED,
        "device": str(device),
    }])
    summary.to_csv(RESULT_DIR / "summary_gcn_lstm_msefair.csv", index=False, encoding="utf-8-sig")
    with open(RESULT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "graph_cache": str(GRAPH_CACHE),
            "lookback": LOOKBACK,
            "gcn_hidden": GCN_HIDDEN,
            "lstm_hidden": LSTM_HIDDEN,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        }, f, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
