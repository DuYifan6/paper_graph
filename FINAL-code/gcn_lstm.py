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
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn

ROOT = Path(r"D:\PythonProject\LASSO_FINAL")
FAIR_ROOT = ROOT / "AE-3" / "benchmark_fairness"
CACHE_NPZ = ROOT / "common_preprocess_cache" / "common_tensor_cache.npz"
META_PKL = ROOT / "common_preprocess_cache" / "common_tensor_meta.pkl"
# Reuse the already-built positive-only Graph-2 cache.
# Prefer the fairness-project cache actually used by the successful Graph-2 MSE run;
# fall back to the legacy main-graph location if it exists.
GRAPH_CACHE_CANDIDATES = [
    FAIR_ROOT / "graph_cache" / "graph2_positive_roll20_topk40_tau0.1_binary.npz",
    ROOT / "graph_大修" / "graph2_positive_only" / "graphs" / "roll20_binary_positive_mkt1_topk40_shift0_tau0.1_pow1.0_full1.npz",
]

# Also accept a matching cache with a slightly different filename under the fairness graph_cache folder.
if (FAIR_ROOT / "graph_cache").exists():
    GRAPH_CACHE_CANDIDATES.extend(
        sorted((FAIR_ROOT / "graph_cache").glob("*positive*topk40*.npz"))
    )

GRAPH_CACHE = next((p for p in GRAPH_CACHE_CANDIDATES if p.exists()), GRAPH_CACHE_CANDIDATES[0])

OUT_DIR = FAIR_ROOT / "gcn_lstm"
RESULT_DIR = OUT_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 20
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
    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=True)
        self.nbr_idx = z["nbr_idx"]
        self.nbr_w = z["nbr_w"].astype(np.float32)
        self.node_valid = z["node_valid"].astype(bool) if "node_valid" in z.files else None

    def seq(self, t: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Correct alignment: same date range as X[t-L+1:t+1].
        ni = np.asarray(self.nbr_idx[t - LOOKBACK + 1:t + 1], dtype=np.int64)
        nw = np.asarray(self.nbr_w[t - LOOKBACK + 1:t + 1], dtype=np.float32)
        return torch.as_tensor(ni, dtype=torch.long, device=device), torch.as_tensor(nw, dtype=torch.float32, device=device)


def prepare_x_seq(X: np.ndarray, t: int, device: torch.device) -> torch.Tensor:
    # X[t] is already t-1 information; include X[t] as the latest lagged predictor row.
    x = np.asarray(X[t - LOOKBACK + 1:t + 1], dtype=np.float32)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


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
        tried = "\n".join(f"  - {p}" for p in GRAPH_CACHE_CANDIDATES)
        raise FileNotFoundError(
            "Existing positive-only Graph-2 cache not found. Tried:\n"
            + tried
            + "\nUse the same graph cache that was used by the successful Graph-2 MSE run."
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
    train_start, fit_end, val_start, train_end, test_start, test_end = resolve_split(dates, meta)
    print(f"[SPLIT] fit targets: {dates[max(train_start, LOOKBACK - 1)].date()} .. {dates[fit_end].date()}")
    print(f"[SPLIT] validation: {dates[val_start].date()} .. {dates[train_end].date()}")
    print(f"[SPLIT] test: {dates[test_start].date()} .. {dates[test_end].date()}")

    fit_days = np.arange(max(LOOKBACK - 1, train_start), fit_end + 1, dtype=np.int32)
    val_days = np.arange(max(LOOKBACK - 1, val_start), train_end + 1, dtype=np.int32)
    test_days = np.arange(max(LOOKBACK - 1, test_start), test_end + 1, dtype=np.int32)

    graph = GraphStore(GRAPH_CACHE)
    if graph.nbr_idx.shape[:2] != X.shape[:2]:
        raise RuntimeError(f"Graph/data mismatch: graph={graph.nbr_idx.shape}, X={X.shape}")

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
