#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOTA baseline experiments for ESWA revision.

Core design:
- Representative baseline set: Ridge, LightGBM, MLP, Transformer, GCN-LSTM.
- Same target, split, feature set, and information timing for all baselines.
- Firm-level models use only lagged daily/15-minute stable predictors.
- Graph baselines use the lagged A_{t-1} graph, but 5-minute returns are NOT appended as features.

Expected tensor cache:
    X.npy             shape [T, N, F], firm-level features by day/stock/feature
    y.npy             shape [T, N], target overnight gap on day t
    valid_mask.npy    shape [T, N], True where y and X are valid
    dates.npy         shape [T], date strings or numpy datetime64
    stocks.npy        shape [N], stock identifiers
    graph2_edges.npz  optional, for graph baselines. See GraphEdgeStore below.

If you already have tensors from your RealGraph/ZeroGraph training code, use them directly.
If not, create them first from your panel file and save as the above arrays.

Example:
python run_sota_baselines.py \
  --data_dir ./tensor_cache_noST_no9pct \
  --graph_edges ./graph2_edges.npz \
  --out_dir ./sota_results \
  --models ridge,lightgbm,mlp,transformer,gcn_lstm

Author: revise and adapt paths/keys before running.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, norm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# =========================
# 1. Configuration
# =========================

@dataclass
class Config:
    data_dir: str
    out_dir: str = "./sota_results"
    graph_edges: Optional[str] = None

    # Date split: target day t belongs to train/test according to dates[t].
    train_start: str = "2022-01-01"
    train_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"

    lookback: int = 20
    seed: int = 2026

    # Modeling.
    # Representative SOTA baselines for ESWA revision.
    # Keep the table concise: linear, tree ensemble, nonlinear NN, temporal attention,
    # and conventional graph-learning baseline.
    models: str = "ridge,lightgbm,mlp,transformer,gcn_lstm"

    # Tree baselines may be heavy on 2M+ rows; set None/0 for full sample.
    tree_max_train_rows: int = 800_000
    linear_max_train_rows: int = 0

    # Neural training.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4096
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 5e-5
    hidden: int = 128
    dropout: float = 0.10
    patience: int = 4
    grad_clip: float = 1.0

    # Graph neural baselines.
    graph_epochs: int = 15
    graph_hidden: int = 128
    gat_heads: int = 4
    graph_batch_days: int = 1  # keep 1 for memory safety
    max_edges_per_day: int = 400_000

    # Evaluation.
    dm_lag: int = 2
    save_predictions: bool = True


# =========================
# 2. Reproducibility
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism may reduce speed, but helps revision reproducibility.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =========================
# 3. Loading tensor cache
# =========================

def load_tensor_cache(data_dir: str) -> Dict[str, np.ndarray]:
    data_dir = Path(data_dir)
    required = ["X.npy", "y.npy", "valid_mask.npy", "dates.npy", "stocks.npy"]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {data_dir}: {missing}\n"
            "Expected X.npy [T,N,F], y.npy [T,N], valid_mask.npy [T,N], dates.npy [T], stocks.npy [N]."
        )

    X = np.load(data_dir / "X.npy", mmap_mode="r")
    y = np.load(data_dir / "y.npy", mmap_mode="r")
    valid = np.load(data_dir / "valid_mask.npy", mmap_mode="r").astype(bool)
    dates = np.load(data_dir / "dates.npy", allow_pickle=True)
    stocks = np.load(data_dir / "stocks.npy", allow_pickle=True)

    dates = pd.to_datetime(dates.astype(str))
    assert X.ndim == 3, f"X must be [T,N,F], got {X.shape}"
    assert y.shape == X.shape[:2], f"y must be [T,N], got {y.shape}, X={X.shape}"
    assert valid.shape == y.shape, f"valid_mask must be [T,N], got {valid.shape}"

    return {"X": X, "y": y, "valid": valid, "dates": dates, "stocks": stocks}


def make_target_indices(dates: pd.DatetimeIndex, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    train_mask = (dates >= pd.Timestamp(cfg.train_start)) & (dates <= pd.Timestamp(cfg.train_end))
    test_mask = (dates >= pd.Timestamp(cfg.test_start)) & (dates <= pd.Timestamp(cfg.test_end))

    # target day t must have lookback days t-L...t-1 available
    t_all = np.arange(len(dates))
    enough_history = t_all >= cfg.lookback

    train_t = t_all[train_mask & enough_history]
    test_t = t_all[test_mask & enough_history]
    if len(train_t) == 0 or len(test_t) == 0:
        raise ValueError(f"Empty train/test target indices. train={len(train_t)}, test={len(test_t)}")
    return train_t, test_t


def sequence_valid_mask(valid: np.ndarray, target_t: np.ndarray, L: int) -> Dict[int, np.ndarray]:
    """
    For each target day t, a stock is valid when:
      - y[t, i] valid
      - all X[t-L:t, i, :] valid according to valid mask.
    If your X-valid and y-valid are separate, replace this with stricter masks.
    """
    out: Dict[int, np.ndarray] = {}
    for t in target_t:
        hist_ok = valid[t - L:t].all(axis=0)
        out[int(t)] = valid[t] & hist_ok
    return out


# =========================
# 4. Flattened dataset for linear/tree/MLP
# =========================

def build_flat_xy(
    X: np.ndarray,
    y: np.ndarray,
    valid_by_t: Dict[int, np.ndarray],
    target_t: np.ndarray,
    L: int,
    max_rows: int = 0,
    seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X_flat: [num_obs, L*F]
      y_flat: [num_obs]
      t_idx:  [num_obs], target date index
      n_idx:  [num_obs], stock index
    """
    rows_x, rows_y, rows_t, rows_n = [], [], [], []
    for t in target_t:
        mask = valid_by_t[int(t)]
        n_idx = np.where(mask)[0]
        if len(n_idx) == 0:
            continue
        seq = np.asarray(X[t - L:t, n_idx, :], dtype=np.float32)  # [L, M, F]
        seq = np.transpose(seq, (1, 0, 2)).reshape(len(n_idx), -1)  # [M, L*F]
        yy = np.asarray(y[t, n_idx], dtype=np.float32)

        rows_x.append(seq)
        rows_y.append(yy)
        rows_t.append(np.full(len(n_idx), t, dtype=np.int32))
        rows_n.append(n_idx.astype(np.int32))

    X_flat = np.vstack(rows_x).astype(np.float32)
    y_flat = np.concatenate(rows_y).astype(np.float32)
    t_idx = np.concatenate(rows_t)
    n_idx = np.concatenate(rows_n)

    if max_rows and max_rows > 0 and len(y_flat) > max_rows:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(y_flat), size=max_rows, replace=False)
        keep.sort()
        X_flat, y_flat, t_idx, n_idx = X_flat[keep], y_flat[keep], t_idx[keep], n_idx[keep]

    return X_flat, y_flat, t_idx, n_idx


class SeqDataset(Dataset):
    def __init__(self, X_flat_or_seq: np.ndarray, y: np.ndarray, as_sequence: bool, L: int, F: int):
        self.X = X_flat_or_seq.astype(np.float32)
        self.y = y.astype(np.float32)
        self.as_sequence = as_sequence
        self.L = L
        self.F = F

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.as_sequence:
            x = x.reshape(self.L, self.F)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.float32)


# =========================
# 5. Metrics and DM test
# =========================

def period_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) < 2:
        return np.nan
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.nan if sst <= 1e-12 else 1.0 - sse / sst


def rank_ic_by_day(df: pd.DataFrame) -> Tuple[float, float]:
    vals = []
    for _, g in df.groupby("date"):
        if g["y_true"].nunique() < 3 or g["y_pred"].nunique() < 3:
            continue
        rho = spearmanr(g["y_true"].values, g["y_pred"].values).correlation
        if np.isfinite(rho):
            vals.append(rho)
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan


def newey_west_se(x: np.ndarray, lag: int = 2) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T <= lag + 1:
        return np.nan
    xc = x - x.mean()
    gamma0 = np.sum(xc * xc) / T
    var = gamma0
    for l in range(1, lag + 1):
        gamma = np.sum(xc[l:] * xc[:-l]) / T
        weight = 1.0 - l / (lag + 1.0)
        var += 2.0 * weight * gamma
    return math.sqrt(max(var, 0.0) / T)


def dm_test_against_baseline(
    pred_df: pd.DataFrame,
    base_df: pd.DataFrame,
    lag: int = 2,
) -> Tuple[float, float, float]:
    """
    Daily d_t = MSE_baseline(t) - MSE_model(t).
    H1: mean(d_t) > 0 means model is better than baseline.
    Returns mean_d, DM_stat, one-sided p.
    """
    merged = pred_df[["date", "stock", "y_true", "y_pred"]].merge(
        base_df[["date", "stock", "y_pred"]],
        on=["date", "stock"],
        suffixes=("", "_base"),
        how="inner",
    )
    ds = []
    for _, g in merged.groupby("date"):
        e_model = (g["y_true"].values - g["y_pred"].values) ** 2
        e_base = (g["y_true"].values - g["y_pred_base"].values) ** 2
        ds.append(float(np.mean(e_base - e_model)))
    d = np.asarray(ds)
    mean_d = float(np.mean(d))
    se = newey_west_se(d, lag=lag)
    dm = mean_d / se if se and np.isfinite(se) and se > 0 else np.nan
    p = 1.0 - norm.cdf(dm) if np.isfinite(dm) else np.nan
    return mean_d, float(dm), float(p)


def summarize_predictions(
    pred_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    model_name: str,
    base_df: Optional[pd.DataFrame] = None,
    dm_lag: int = 2,
) -> Dict[str, float]:
    y_true = pred_df["y_true"].values
    y_pred = pred_df["y_pred"].values

    overall = {
        "model": model_name,
        "n_obs": int(len(pred_df)),
        "oos_r2_all": period_r2(y_true, y_pred),
        "mse_all": float(mean_squared_error(y_true, y_pred)),
        "mae_all": float(mean_absolute_error(y_true, y_pred)),
    }

    ric, ric_se = rank_ic_by_day(pred_df)
    overall["rank_ic_daily_mean"] = ric
    overall["rank_ic_daily_se"] = ric_se

    pred_df = pred_df.copy()
    pred_df["month"] = pd.to_datetime(pred_df["date"]).dt.to_period("M").astype(str)

    monthly = []
    for m, g in pred_df.groupby("month"):
        monthly.append({
            "month": m,
            "model": model_name,
            "n_obs": int(len(g)),
            "r2": period_r2(g["y_true"].values, g["y_pred"].values),
            "mse": float(mean_squared_error(g["y_true"].values, g["y_pred"].values)),
            "mae": float(mean_absolute_error(g["y_true"].values, g["y_pred"].values)),
            "rank_ic": rank_ic_by_day(g)[0],
        })
    monthly_df = pd.DataFrame(monthly)

    if len(monthly_df) > 0:
        # Observation-weighted monthly R2, as commonly used in your current tables.
        w = monthly_df["n_obs"].values / monthly_df["n_obs"].sum()
        overall["weighted_monthly_r2"] = float(np.nansum(w * monthly_df["r2"].values))
        overall["weighted_monthly_mse"] = float(np.nansum(w * monthly_df["mse"].values))
        overall["weighted_monthly_mae"] = float(np.nansum(w * monthly_df["mae"].values))
    else:
        overall["weighted_monthly_r2"] = np.nan
        overall["weighted_monthly_mse"] = np.nan
        overall["weighted_monthly_mae"] = np.nan

    if base_df is not None:
        mean_d, dm, p = dm_test_against_baseline(pred_df, base_df, lag=dm_lag)
        overall["dm_mean_d_vs_base"] = mean_d
        overall["dm_stat_vs_base"] = dm
        overall["dm_p_one_sided_vs_base"] = p

    return overall, monthly_df


# =========================
# 6. sklearn baselines
# =========================

def fit_predict_sklearn(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    name = model_name.lower()
    t0 = time.time()

    if name == "ridge":
        model = Ridge(alpha=10.0, random_state=cfg.seed)
    elif name == "elasticnet":
        model = ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000, random_state=cfg.seed)
    elif name == "lasso":
        model = Lasso(alpha=1e-5, max_iter=5000, random_state=cfg.seed)
    elif name == "rf":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=cfg.seed,
            verbose=0,
        )
    elif name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("Install lightgbm first: pip install lightgbm") from e
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=cfg.seed,
            n_jobs=-1,
        )
    elif name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("Install xgboost first: pip install xgboost") from e
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_lambda=1.0,
            reg_alpha=0.0,
            tree_method="hist",
            random_state=cfg.seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown sklearn model: {model_name}")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    model.fit(Xtr, y_train)
    pred = model.predict(Xte).astype(np.float32)

    print(f"[{model_name}] done in {time.time() - t0:.1f}s")
    return pred


# =========================
# 7. Neural non-graph baselines
# =========================

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        return self.net(x).squeeze(-1)


class RNNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, kind: str = "lstm", dropout: float = 0.1):
        super().__init__()
        rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_dim, hidden, num_layers=1, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        h = out[:, -1, :]
        return self.head(h).squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.down(x)


class TCNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.tcn = nn.Sequential(
            TemporalBlock(input_dim, hidden, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock(hidden, hidden, kernel_size=3, dilation=2, dropout=dropout),
            TemporalBlock(hidden, hidden, kernel_size=3, dilation=4, dropout=dropout),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x [B,L,F] -> [B,F,L]
        z = self.tcn(x.transpose(1, 2))
        h = z[:, :, -1]
        return self.head(h).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.1, nhead: int = 4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.pe = PositionalEncoding(hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        z = self.proj(x)
        z = self.pe(z)
        z = self.encoder(z)
        h = z[:, -1, :]
        return self.head(h).squeeze(-1)


def make_neural_model(name: str, L: int, F: int, cfg: Config) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLPRegressor(L * F, cfg.hidden, cfg.dropout)
    if name == "lstm":
        return RNNRegressor(F, cfg.hidden, kind="lstm", dropout=cfg.dropout)
    if name == "gru":
        return RNNRegressor(F, cfg.hidden, kind="gru", dropout=cfg.dropout)
    if name == "tcn":
        return TCNRegressor(F, cfg.hidden, cfg.dropout)
    if name == "transformer":
        return TransformerRegressor(F, cfg.hidden, cfg.dropout, nhead=4)
    raise ValueError(f"Unknown neural model: {name}")


def train_predict_neural(
    model_name: str,
    X_train_flat: np.ndarray,
    y_train: np.ndarray,
    X_test_flat: np.ndarray,
    cfg: Config,
    F: int,
) -> np.ndarray:
    as_seq = model_name.lower() != "mlp"
    L = cfg.lookback
    model = make_neural_model(model_name, L, F, cfg).to(cfg.device)

    # Standardize flattened features, then reshape for sequence models.
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train_flat).astype(np.float32)
    Xte = scaler.transform(X_test_flat).astype(np.float32)

    ds = SeqDataset(Xtr, y_train, as_sequence=as_seq, L=L, F=F)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_loss = np.inf
    best_state = None
    bad = 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for xb, yb in dl:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        ep_loss = float(np.mean(losses))
        print(f"[{model_name}] epoch {ep:02d}/{cfg.epochs} train_mse={ep_loss:.6f}")

        if ep_loss < best_loss - 1e-6:
            best_loss = ep_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    test_ds = SeqDataset(Xte, np.zeros(len(Xte), dtype=np.float32), as_sequence=as_seq, L=L, F=F)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for xb, _ in test_dl:
            pred = model(xb.to(cfg.device)).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds).astype(np.float32)


# =========================
# 8. Graph baseline support
# =========================

class GraphEdgeStore:
    """
    Expected graph_edges.npz formats supported:

    Format A: object arrays
        edge_index_by_t: object array length T, each item shape [2,E]
        edge_weight_by_t: optional object array length T, each item shape [E]

    Format B: CSR-like arrays for each t
        indices_by_t, indptr_by_t, data_by_t as object arrays.
        Each t defines scipy CSR rows -> columns.

    Format C: dense adjacency object array
        adj_by_t: object array length T, each item [N,N] or scipy sparse-like unsupported here.

    The graph for target day t should be A_{t-1}. Therefore get_edges(t) loads t-1 by default.
    """
    def __init__(self, path: str, num_nodes: int, device: str, max_edges_per_day: int = 400_000):
        self.path = path
        self.num_nodes = num_nodes
        self.device = device
        self.max_edges_per_day = max_edges_per_day
        self.z = np.load(path, allow_pickle=True)
        self.keys = set(self.z.files)

    def get_edges(self, target_t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        graph_t = target_t - 1

        if "edge_index_by_t" in self.keys:
            ei = self.z["edge_index_by_t"][graph_t]
            ew = self.z["edge_weight_by_t"][graph_t] if "edge_weight_by_t" in self.keys else np.ones(ei.shape[1], dtype=np.float32)
        elif {"indices_by_t", "indptr_by_t"}.issubset(self.keys):
            indices = self.z["indices_by_t"][graph_t]
            indptr = self.z["indptr_by_t"][graph_t]
            data = self.z["data_by_t"][graph_t] if "data_by_t" in self.keys else None
            rows = []
            cols = []
            vals = []
            for r in range(len(indptr) - 1):
                start, end = indptr[r], indptr[r + 1]
                if end > start:
                    c = indices[start:end]
                    rows.append(np.full(len(c), r, dtype=np.int64))
                    cols.append(c.astype(np.int64))
                    vals.append(data[start:end].astype(np.float32) if data is not None else np.ones(len(c), dtype=np.float32))
            if rows:
                ei = np.vstack([np.concatenate(rows), np.concatenate(cols)])
                ew = np.concatenate(vals)
            else:
                ei = np.zeros((2, 0), dtype=np.int64)
                ew = np.zeros(0, dtype=np.float32)
        elif "adj_by_t" in self.keys:
            A = self.z["adj_by_t"][graph_t]
            rr, cc = np.nonzero(A)
            ei = np.vstack([rr, cc]).astype(np.int64)
            ew = A[rr, cc].astype(np.float32)
        else:
            raise KeyError(
                f"Unsupported graph npz keys={self.keys}. "
                "Please save edge_index_by_t/edge_weight_by_t or CSR object arrays."
            )

        if ei.shape[1] > self.max_edges_per_day:
            # Keep top weights or random sample if all binary.
            rng = np.random.default_rng(12345 + int(target_t))
            keep = rng.choice(ei.shape[1], size=self.max_edges_per_day, replace=False)
            ei = ei[:, keep]
            ew = ew[keep]

        edge_index = torch.as_tensor(ei, dtype=torch.long, device=self.device)
        edge_weight = torch.as_tensor(ew, dtype=torch.float32, device=self.device)
        return edge_index, edge_weight


def add_self_loops(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    device = edge_index.device
    loop = torch.arange(num_nodes, device=device, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    loop_w = torch.ones(num_nodes, device=device, dtype=torch.float32)
    return torch.cat([edge_index, loop], dim=1), torch.cat([edge_weight, loop_w], dim=0)


def gcn_norm(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, device=edge_index.device).scatter_add_(0, row, edge_weight)
    deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
    norm_w = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, norm_w


def aggregate(edge_index: torch.Tensor, edge_weight: torch.Tensor, x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Sparse weighted aggregation: out[row] += w * x[col].
    edge_index[0]=target row, edge_index[1]=source col.
    """
    row, col = edge_index[0], edge_index[1]
    msg = x[col] * edge_weight.unsqueeze(-1)
    out = torch.zeros_like(x)
    out.index_add_(0, row, msg)
    return out


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        ei, ew = gcn_norm(edge_index, edge_weight, x.shape[0])
        out = aggregate(ei, ew, x, x.shape[0])
        return self.lin(out)


class SimpleGATLayer(nn.Module):
    """
    Lightweight GAT implemented without torch_geometric.
    For large graphs this is slower than PyG, but avoids extra dependency.
    """
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.heads = heads
        self.dh = out_dim // heads
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.att_l = nn.Parameter(torch.empty(heads, self.dh))
        self.att_r = nn.Parameter(torch.empty(heads, self.dh))
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, edge_weight):
        N = x.shape[0]
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, N)
        row, col = edge_index[0], edge_index[1]
        h = self.lin(x).view(N, self.heads, self.dh)
        e = (h[row] * self.att_l).sum(-1) + (h[col] * self.att_r).sum(-1)
        e = torch.nn.functional.leaky_relu(e, negative_slope=0.2)

        # Softmax over incoming edges per row. Implemented per head.
        alpha_all = []
        for k in range(self.heads):
            ek = e[:, k]
            max_per_row = torch.full((N,), -1e30, device=x.device)
            max_per_row.scatter_reduce_(0, row, ek, reduce="amax", include_self=True)
            exp = torch.exp(ek - max_per_row[row]) * edge_weight.clamp(min=0).sqrt()
            denom = torch.zeros(N, device=x.device).scatter_add_(0, row, exp).clamp(min=1e-12)
            alpha_all.append(exp / denom[row])
        alpha = torch.stack(alpha_all, dim=1)
        alpha = self.dropout(alpha)

        msg = h[col] * alpha.unsqueeze(-1)
        out = torch.zeros((N, self.heads, self.dh), device=x.device)
        # index_add for each head
        out.index_add_(0, row, msg)
        return out.reshape(N, self.heads * self.dh) + self.bias


class GraphLSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int, graph_type: str = "gcn", gat_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
        if graph_type == "gcn":
            self.graph = GCNLayer(hidden, hidden)
        elif graph_type == "gat":
            self.graph = SimpleGATLayer(hidden, hidden, heads=gat_heads, dropout=dropout)
        else:
            raise ValueError(graph_type)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_seq_all_nodes, edge_index, edge_weight):
        """
        x_seq_all_nodes: [N,L,F]
        """
        out, _ = self.rnn(x_seq_all_nodes)
        h = out[:, -1, :]
        hg = self.graph(h, edge_index, edge_weight)
        # Conventional graph baseline: graph transformation directly changes representation.
        z = torch.relu(hg)
        return self.head(z).squeeze(-1)


def train_predict_graph_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    valid_by_t_train: Dict[int, np.ndarray],
    valid_by_t_test: Dict[int, np.ndarray],
    train_t: np.ndarray,
    test_t: np.ndarray,
    dates: pd.DatetimeIndex,
    stocks: np.ndarray,
    cfg: Config,
) -> pd.DataFrame:
    if cfg.graph_edges is None:
        raise ValueError("graph_edges is required for gcn_lstm/gat_lstm.")

    T, N, F = X.shape
    graph_store = GraphEdgeStore(cfg.graph_edges, num_nodes=N, device=cfg.device, max_edges_per_day=cfg.max_edges_per_day)
    graph_type = "gcn" if model_name.lower() == "gcn_lstm" else "gat"

    model = GraphLSTMRegressor(
        input_dim=F,
        hidden=cfg.graph_hidden,
        graph_type=graph_type,
        gat_heads=cfg.gat_heads,
        dropout=cfg.dropout,
    ).to(cfg.device)

    # Standardize features by train target sequences only.
    # For graph daily training, transform X[t-L:t,:,:] using feature-wise train mean/std.
    train_vals = []
    for t in train_t:
        mask = valid_by_t_train[int(t)]
        if mask.any():
            train_vals.append(np.asarray(X[t - cfg.lookback:t, mask, :]).reshape(-1, F))
    train_vals = np.vstack(train_vals)
    mu = np.nanmean(train_vals, axis=0).astype(np.float32)
    sd = np.nanstd(train_vals, axis=0).astype(np.float32)
    sd[sd < 1e-8] = 1.0

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_loss = np.inf
    best_state = None
    bad = 0

    for ep in range(1, cfg.graph_epochs + 1):
        model.train()
        day_order = list(map(int, train_t))
        random.shuffle(day_order)
        losses = []

        for t in day_order:
            mask_np = valid_by_t_train[t]
            if mask_np.sum() < 50:
                continue

            seq_np = np.asarray(X[t - cfg.lookback:t, :, :], dtype=np.float32)  # [L,N,F]
            seq_np = (seq_np - mu) / sd
            seq_np = np.nan_to_num(seq_np, nan=0.0, posinf=0.0, neginf=0.0)
            seq = torch.as_tensor(np.transpose(seq_np, (1, 0, 2)), dtype=torch.float32, device=cfg.device)  # [N,L,F]

            yy = torch.as_tensor(np.asarray(y[t, :], dtype=np.float32), device=cfg.device)
            mask = torch.as_tensor(mask_np, dtype=torch.bool, device=cfg.device)

            edge_index, edge_weight = graph_store.get_edges(t)

            opt.zero_grad(set_to_none=True)
            pred_all = model(seq, edge_index, edge_weight)
            loss = loss_fn(pred_all[mask], yy[mask])
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        ep_loss = float(np.mean(losses)) if losses else np.nan
        print(f"[{model_name}] epoch {ep:02d}/{cfg.graph_epochs} train_mse={ep_loss:.6f}")

        if np.isfinite(ep_loss) and ep_loss < best_loss - 1e-6:
            best_loss = ep_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict test by daily full graph.
    model.eval()
    rows = []
    with torch.no_grad():
        for t in map(int, test_t):
            mask_np = valid_by_t_test[t]
            if mask_np.sum() == 0:
                continue
            seq_np = np.asarray(X[t - cfg.lookback:t, :, :], dtype=np.float32)
            seq_np = (seq_np - mu) / sd
            seq_np = np.nan_to_num(seq_np, nan=0.0, posinf=0.0, neginf=0.0)
            seq = torch.as_tensor(np.transpose(seq_np, (1, 0, 2)), dtype=torch.float32, device=cfg.device)

            edge_index, edge_weight = graph_store.get_edges(t)
            pred_all = model(seq, edge_index, edge_weight).detach().cpu().numpy()
            idx = np.where(mask_np)[0]
            rows.append(pd.DataFrame({
                "date": str(pd.Timestamp(dates[t]).date()),
                "t_idx": t,
                "stock": stocks[idx].astype(str),
                "n_idx": idx,
                "y_true": np.asarray(y[t, idx], dtype=np.float32),
                "y_pred": pred_all[idx].astype(np.float32),
            }))

    return pd.concat(rows, ignore_index=True)


# =========================
# 9. Main runner
# =========================

def make_prediction_df_from_flat(
    y_test: np.ndarray,
    pred: np.ndarray,
    t_idx: np.ndarray,
    n_idx: np.ndarray,
    dates: pd.DatetimeIndex,
    stocks: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame({
        "date": [str(pd.Timestamp(dates[int(t)]).date()) for t in t_idx],
        "t_idx": t_idx.astype(int),
        "stock": stocks[n_idx].astype(str),
        "n_idx": n_idx.astype(int),
        "y_true": y_test.astype(np.float32),
        "y_pred": pred.astype(np.float32),
    })


def run(cfg: Config) -> None:
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    cache = load_tensor_cache(cfg.data_dir)
    X, y, valid, dates, stocks = cache["X"], cache["y"], cache["valid"], cache["dates"], cache["stocks"]
    T, N, F = X.shape
    print(f"Loaded X={X.shape}, y={y.shape}, dates={dates[0]}..{dates[-1]}, N={N}, F={F}")

    train_t, test_t = make_target_indices(dates, cfg)
    print(f"Train target days: {len(train_t)}, {dates[train_t[0]]}..{dates[train_t[-1]]}")
    print(f"Test target days:  {len(test_t)}, {dates[test_t[0]]}..{dates[test_t[-1]]}")

    valid_train = sequence_valid_mask(valid, train_t, cfg.lookback)
    valid_test = sequence_valid_mask(valid, test_t, cfg.lookback)

    # Build flat train/test once for non-graph models.
    lin_max = cfg.linear_max_train_rows if cfg.linear_max_train_rows else 0
    X_train_full, y_train_full, _, _ = build_flat_xy(
        X, y, valid_train, train_t, cfg.lookback, max_rows=lin_max, seed=cfg.seed
    )
    X_test_flat, y_test_flat, t_test_flat, n_test_flat = build_flat_xy(
        X, y, valid_test, test_t, cfg.lookback, max_rows=0, seed=cfg.seed
    )
    print(f"Flat train={X_train_full.shape}, test={X_test_flat.shape}")

    requested = [m.strip().lower() for m in cfg.models.split(",") if m.strip()]
    summary_rows = []
    monthly_all = []
    pred_dfs: Dict[str, pd.DataFrame] = {}

    # Only run models explicitly specified by --models.
    # If you want LSTM/ZeroGraph as an additional baseline, include lstm manually:
    #   --models lstm,ridge,lightgbm,mlp,transformer
    requested_for_base = requested

    for model_name in requested_for_base:
        print("\n" + "=" * 80)
        print(f"Running model: {model_name}")
        t0 = time.time()

        if model_name in {"ridge", "elasticnet", "lasso"}:
            max_rows = cfg.linear_max_train_rows if cfg.linear_max_train_rows else 0
            Xtr, ytr = X_train_full, y_train_full
            if max_rows and len(ytr) > max_rows:
                Xtr, ytr, _, _ = build_flat_xy(X, y, valid_train, train_t, cfg.lookback, max_rows=max_rows, seed=cfg.seed)
            pred = fit_predict_sklearn(model_name, Xtr, ytr, X_test_flat, cfg)
            pred_df = make_prediction_df_from_flat(y_test_flat, pred, t_test_flat, n_test_flat, dates, stocks)

        elif model_name in {"rf", "lightgbm", "xgboost"}:
            max_rows = cfg.tree_max_train_rows if cfg.tree_max_train_rows else 0
            Xtr, ytr, _, _ = build_flat_xy(X, y, valid_train, train_t, cfg.lookback, max_rows=max_rows, seed=cfg.seed)
            pred = fit_predict_sklearn(model_name, Xtr, ytr, X_test_flat, cfg)
            pred_df = make_prediction_df_from_flat(y_test_flat, pred, t_test_flat, n_test_flat, dates, stocks)

        elif model_name in {"mlp", "lstm", "gru", "tcn", "transformer"}:
            pred = train_predict_neural(model_name, X_train_full, y_train_full, X_test_flat, cfg, F=F)
            pred_df = make_prediction_df_from_flat(y_test_flat, pred, t_test_flat, n_test_flat, dates, stocks)

        elif model_name in {"gcn_lstm", "gat_lstm"}:
            pred_df = train_predict_graph_model(
                model_name, X, y, valid_train, valid_test, train_t, test_t, dates, stocks, cfg
            )
        else:
            print(f"Skip unknown model: {model_name}")
            continue

        elapsed = time.time() - t0
        pred_df["model"] = model_name
        pred_dfs[model_name] = pred_df

        if cfg.save_predictions:
            # 1) 保存 parquet，供原脚本继续使用
            pred_df.to_parquet(out_dir / f"pred_{model_name}.parquet", index=False)

            # 2) 额外保存 CSV 明细，方便 DM test 和论文表格统一读取
            detail_path = out_dir / f"pred_{model_name}_detail.csv"
            pred_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

            print(f"Saved prediction detail: {detail_path}, rows={len(pred_df):,}")
            print(pred_df.head())

        base_df = pred_dfs.get("lstm")
        base_for_eval = None if model_name == "lstm" or base_df is None else base_df
        summ, monthly = summarize_predictions(pred_df, dates, model_name, base_df=base_for_eval, dm_lag=cfg.dm_lag)
        summ["elapsed_sec"] = elapsed
        summary_rows.append(summ)
        monthly_all.append(monthly)

        pd.DataFrame(summary_rows).to_csv(out_dir / "sota_summary_running.csv", index=False, encoding="utf-8-sig")
        print(pd.Series(summ).to_string())

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "sota_summary.csv", index=False, encoding="utf-8-sig")

    if monthly_all:
        monthly_df = pd.concat(monthly_all, ignore_index=True)
        monthly_df.to_csv(out_dir / "sota_monthly_metrics.csv", index=False, encoding="utf-8-sig")

    print("\nFinished. Outputs:")
    print(f"- {out_dir / 'sota_summary.csv'}")
    print(f"- {out_dir / 'sota_monthly_metrics.csv'}")
    print(f"- pred_*.parquet if save_predictions=True")


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="./sota_results")
    p.add_argument("--graph_edges", default=None)
    p.add_argument("--models", default=Config.models)
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--train_start", default="2022-01-01")
    p.add_argument("--train_end", default="2024-12-31")
    p.add_argument("--test_start", default="2025-01-01")
    p.add_argument("--test_end", default="2025-12-31")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--graph_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--tree_max_train_rows", type=int, default=800000)
    p.add_argument("--linear_max_train_rows", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    return Config(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
