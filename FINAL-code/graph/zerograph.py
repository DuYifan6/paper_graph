# -*- coding: utf-8 -*-
"""
ZeroGraph for the ESWA common-MSE fairness audit.

Fairness rules:
- reads the ORIGINAL common_tensor_cache used by the main experiments
- same chronological fitting / validation / 2025 test periods for every fairness model
- 20-day lagged input window ending at X[t]
  (X[t] already contains predictors from t-1 in the raw panel)
- training loss: MSE only
- checkpoint selection: minimum observation-weighted validation MSE
- early stopping: patience=4
- no ranking loss and no Top-K checkpoint rule
"""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(r"D:\PythonProject\LASSO_FINAL")
FAIR_ROOT = ROOT / "AE-3" / "benchmark_fairness"
CACHE_NPZ = ROOT / "common_preprocess_cache" / "common_tensor_cache.npz"
META_PKL = ROOT / "common_preprocess_cache" / "common_tensor_meta.pkl"
OUT_DIR = FAIR_ROOT / "zerograph_msefair"
RESULT_DIR = OUT_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

EXP_NAME = "ZeroGraph_MSEFair"
LOOKBACK = 20
HIDDEN = 128
NUM_LAYERS = 1
DROPOUT = 0.10
GATE_HIDDEN = 128
LR = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
GRAD_CLIP = 1.0
PATIENCE = 4
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


class GraphMixKNN(nn.Module):
    def __init__(self, hidden: int, gate_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.lin_gate = nn.Linear(hidden * 2, gate_hidden)
        self.lin_gate2 = nn.Linear(gate_hidden, hidden)
        self.lin_msg = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Z, nbr_idx=None, nbr_w=None, node_valid_mask=None):
        if nbr_idx is None or nbr_w is None:
            return Z
        N, H = Z.shape
        K = nbr_idx.shape[1]
        Z_send = Z * node_valid_mask.view(N, 1) if node_valid_mask is not None else Z
        w0 = torch.nan_to_num(nbr_w, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        Z_nb_raw = Z_send.index_select(0, nbr_idx.reshape(-1)).view(N, K, H)
        w = w0.view(N, K, 1)
        Z_nb = (w * Z_nb_raw).sum(dim=1) / (w0.sum(dim=1, keepdim=True) + 1e-6)
        cat = torch.cat([Z, Z_nb], dim=1)
        g = torch.sigmoid(self.lin_gate2(F.relu(self.lin_gate(cat))))
        delta = self.dropout(F.relu(self.lin_msg(cat)))
        return Z + g * delta


class GraphAugLSTM(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=HIDDEN,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=(DROPOUT if NUM_LAYERS > 1 else 0.0),
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.graph = GraphMixKNN(HIDDEN, GATE_HIDDEN, DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, X, nbr_idx=None, nbr_w=None, node_valid_mask=None):
        out, _ = self.lstm(X)
        z = self.dropout(out[:, -1, :])
        z = self.graph(z, nbr_idx=nbr_idx, nbr_w=nbr_w, node_valid_mask=node_valid_mask)
        return self.head(z).squeeze(-1)


def parameter_audit(model: nn.Module) -> Dict[str, int]:
    return {
        "lstm": int(sum(p.numel() for p in model.lstm.parameters() if p.requires_grad)),
        "graphmix_registered": int(sum(p.numel() for p in model.graph.parameters() if p.requires_grad)),
        "head": int(sum(p.numel() for p in model.head.parameters() if p.requires_grad)),
        "total_registered": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def validation_mse(model, X, y, mask, val_start: int, val_end: int) -> float:
    model.eval()
    sse, n = 0.0, 0
    with torch.no_grad():
        for t in range(max(LOOKBACK - 1, val_start), val_end + 1):
            mt = mask[t]
            n_t = int(mt.sum().item())
            if n_t < MIN_VALID_NODES_PER_DAY:
                continue
            seq = X[t - LOOKBACK + 1:t + 1].permute(1, 0, 2).contiguous()
            pred = model(seq, nbr_idx=None, nbr_w=None, node_valid_mask=mt.float())
            err = pred[mt] - y[t][mt]
            if not torch.isfinite(err).all():
                continue
            sse += float(torch.sum(err * err).item())
            n += n_t
    return float(sse / n) if n > 0 else float("inf")


def evaluate(model, X, y, mask, dates, codes, start_idx: int, end_idx: int):
    model.eval()
    rows = []
    with torch.no_grad():
        for t in range(max(LOOKBACK - 1, start_idx), end_idx + 1):
            mt = mask[t]
            if int(mt.sum().item()) < MIN_VALID_NODES_PER_DAY:
                continue
            seq = X[t - LOOKBACK + 1:t + 1].permute(1, 0, 2).contiguous()
            pred = model(seq, nbr_idx=None, nbr_w=None, node_valid_mask=mt.float())
            idx = torch.where(mt)[0].detach().cpu().numpy()
            yp = pred[mt].detach().cpu().numpy()
            yt = y[t][mt].detach().cpu().numpy()
            d = str(pd.Timestamp(dates[t]).date())
            rows.append(pd.DataFrame({
                "date": d,
                "stock": np.asarray(codes, dtype=object)[idx].astype(str),
                "y_true": yt.astype(np.float32),
                "y_pred": yp.astype(np.float32),
                "model": EXP_NAME,
            }))
    if not rows:
        raise RuntimeError("No evaluation observations.")
    df = pd.concat(rows, ignore_index=True)
    mse = float(mean_squared_error(df["y_true"], df["y_pred"]))
    r2 = float(r2_score(df["y_true"], df["y_pred"]))
    return df, mse, r2


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = np.load(CACHE_NPZ)
    X_np = pack["X_np"].astype(np.float32)
    y_np = pack["y_np"].astype(np.float32)
    mask_np = pack["mask_np"].astype(bool)
    with open(META_PKL, "rb") as f:
        meta = pickle.load(f)

    dates = [pd.Timestamp(d).normalize() for d in meta["all_dates"]]
    codes = list(meta["ts_codes_order"])
    train_start, fit_end, val_start, train_end, test_start, test_end = resolve_split(dates, meta)
    print(f"[SPLIT] fit targets: {dates[max(train_start, LOOKBACK - 1)].date()} .. {dates[fit_end].date()}")
    print(f"[SPLIT] validation: {dates[val_start].date()} .. {dates[train_end].date()}")
    print(f"[SPLIT] test: {dates[test_start].date()} .. {dates[test_end].date()}")

    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    mask = torch.from_numpy(mask_np).to(device)

    model = GraphAugLSTM(in_dim=X_np.shape[2]).to(device)
    audit = parameter_audit(model)
    print("[PARAMS]", audit)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    bad = 0
    grad_audit_done = False

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_sse_like, total_n, used_days = 0.0, 0, 0
        for t in range(max(LOOKBACK - 1, train_start), fit_end + 1):
            mt = mask[t]
            n_t = int(mt.sum().item())
            if n_t < MIN_VALID_NODES_PER_DAY:
                continue
            seq = X[t - LOOKBACK + 1:t + 1].permute(1, 0, 2).contiguous()
            pred = model(seq, nbr_idx=None, nbr_w=None, node_valid_mask=mt.float())
            loss = F.mse_loss(pred[mt], y[t][mt])
            opt.zero_grad(set_to_none=True)
            loss.backward()

            if not grad_audit_done:
                graph_grad = sum(
                    p.numel() for name, p in model.named_parameters()
                    if name.startswith("graph.") and p.requires_grad and p.grad is not None
                )
                if graph_grad != 0:
                    raise RuntimeError(f"ZeroGraph GraphMix unexpectedly has gradients: {graph_grad}")
                grad_audit_done = True

            if GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total_sse_like += float(loss.item()) * n_t
            total_n += n_t
            used_days += 1

        train_mse = total_sse_like / max(total_n, 1)
        val_mse = validation_mse(model, X, y, mask, val_start, train_end)
        history.append({"epoch": ep, "train_mse": train_mse, "val_mse": val_mse, "used_days": used_days})
        print(f"epoch {ep:02d}/{EPOCHS} train_mse={train_mse:.8f} val_mse={val_mse:.8f}")

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
    torch.save(best_state, RESULT_DIR / "best_ZeroGraph_MSEFair.pt")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(RESULT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

    val_df, val_mse_check, val_r2 = evaluate(model, X, y, mask, dates, codes, val_start, train_end)
    test_df, test_mse, test_r2 = evaluate(model, X, y, mask, dates, codes, test_start, test_end)
    val_df.to_csv(RESULT_DIR / "pred_zerograph_validation.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(RESULT_DIR / "pred_zerograph_test.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        "model": EXP_NAME,
        "selection_criterion": "minimum validation MSE",
        "training_loss": "MSE",
        "ranking_loss": 0,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "validation_mse_recomputed": val_mse_check,
        "validation_r2": val_r2,
        "test_mse": test_mse,
        "test_r2": test_r2,
        "n_test": len(test_df),
        "lookback": LOOKBACK,
        "seed": SEED,
        **audit,
    }])
    summary.to_csv(RESULT_DIR / "summary_zerograph_msefair.csv", index=False, encoding="utf-8-sig")
    with open(RESULT_DIR / "split_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "fit_start": str(dates[train_start].date()),
            "fit_end": str(dates[fit_end].date()),
            "validation_start": str(dates[val_start].date()),
            "validation_end": str(dates[train_end].date()),
            "test_start": str(dates[test_start].date()),
            "test_end": str(dates[test_end].date()),
        }, f, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
