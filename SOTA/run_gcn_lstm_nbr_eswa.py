#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GCN-LSTM graph SOTA baseline for ESWA revision.

Adapted to your Graph-2 cache:
    dates       [T]
    nbr_idx     [T, N, K]
    nbr_w       [T, N, K]
    node_valid  [T, N]

No torch_geometric required.

Run example on Windows:
D:\Anaconda3\python.exe D:\PythonProject\LASSO_FINAL\SOTA\run_gcn_lstm_nbr_eswa.py ^
  --data_dir "D:\PythonProject\LASSO_FINAL\SOTA\tensor_cache_noST_no9pct" ^
  --graph_cache "D:\PythonProject\LASSO_FINAL\graph_调整\graph2_binary_60\graphs\roll20_binary_abs1_mkt1_topk60_shift0_tau0.1_pow1.5_full1.npz" ^
  --out_dir "D:\PythonProject\LASSO_FINAL\SOTA\sota_results_gcn_lstm" ^
  --epochs 15 --lookback 20 --amp
"""

from __future__ import annotations

# Put this before numpy/torch imports to avoid Windows Anaconda OpenMP crash.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def safe_dates(x) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(x).astype(str))


def load_data(data_dir: str, graph_cache: str):
    data_dir = Path(data_dir)
    X = np.load(data_dir / "X.npy", mmap_mode="r")
    y = np.load(data_dir / "y.npy", mmap_mode="r")
    valid = np.load(data_dir / "valid_mask.npy", mmap_mode="r").astype(bool)
    dates = safe_dates(np.load(data_dir / "dates.npy", allow_pickle=True))
    stocks = np.load(data_dir / "stocks.npy", allow_pickle=True).astype(str)

    g = np.load(graph_cache, allow_pickle=True)
    if "nbr_idx" not in g.files or "nbr_w" not in g.files:
        raise KeyError(f"Graph cache must contain nbr_idx and nbr_w. keys={g.files}")
    nbr_idx = g["nbr_idx"]
    nbr_w = g["nbr_w"]
    node_valid = g["node_valid"].astype(bool) if "node_valid" in g.files else np.ones(y.shape, dtype=bool)

    assert X.shape[:2] == y.shape == valid.shape
    assert nbr_idx.shape[:2] == y.shape
    assert nbr_w.shape == nbr_idx.shape
    assert node_valid.shape == y.shape
    assert len(dates) == X.shape[0]
    assert len(stocks) == X.shape[1]
    return X, y, valid, dates, stocks, nbr_idx, nbr_w, node_valid


def make_target_days(dates, lookback, train_start, train_end, test_start, test_end, valid_ratio):
    all_t = np.arange(len(dates))
    enough = all_t >= lookback
    train_full = all_t[(dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end)) & enough]
    test_t = all_t[(dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end)) & enough]
    if len(train_full) < 30 or len(test_t) == 0:
        raise ValueError(f"Bad split: train_full={len(train_full)}, test={len(test_t)}")
    n_val = max(5, int(len(train_full) * valid_ratio))
    return train_full[:-n_val].astype(int), train_full[-n_val:].astype(int), test_t.astype(int)


def valid_nodes_for_t(valid, node_valid, t: int, lookback: int, min_nodes: int):
    m = valid[t] & valid[t - lookback:t].all(axis=0)
    m = m & node_valid[t - lookback:t].all(axis=0)
    return m if int(m.sum()) >= min_nodes else None


def calc_feature_scaler(X, valid, train_t, lookback):
    vals = []
    F = X.shape[2]
    for t in map(int, train_t):
        m = valid[t] & valid[t - lookback:t].all(axis=0)
        if m.sum() > 0:
            vals.append(np.asarray(X[t - lookback:t, m, :], dtype=np.float32).reshape(-1, F))
    vals = np.vstack(vals)
    mu = np.nanmean(vals, axis=0).astype(np.float32)
    sd = np.nanstd(vals, axis=0).astype(np.float32)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def prepare_day_tensors(X, nbr_idx, nbr_w, t: int, lookback: int, mu, sd, device):
    x_np = np.asarray(X[t - lookback:t, :, :], dtype=np.float32)
    x_np = (x_np - mu) / sd
    x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
    x_t = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    ni_t = torch.as_tensor(np.asarray(nbr_idx[t - lookback:t], dtype=np.int64), dtype=torch.long, device=device)
    nw_t = torch.as_tensor(np.asarray(nbr_w[t - lookback:t], dtype=np.float32), dtype=torch.float32, device=device)
    return x_t, ni_t, nw_t


class NeighborGCNLayer(nn.Module):
    """h_i = W_self x_i + W_nei sum_j normalized(w_ij) x_j"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.nei_lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, nbr_idx: torch.Tensor, nbr_w: torch.Tensor) -> torch.Tensor:
        neigh = x[nbr_idx]  # [N,K,F]
        w = nbr_w.float().clamp_min(0)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        agg = (neigh * w.unsqueeze(-1)).sum(dim=1)
        h = self.self_lin(x) + self.nei_lin(agg)
        h = torch.relu(self.norm(h))
        return self.dropout(h)


class GCNLSTM(nn.Module):
    def __init__(self, in_dim: int, g_hidden: int = 64, lstm_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.gcn1 = NeighborGCNLayer(in_dim, g_hidden, dropout)
        self.gcn2 = NeighborGCNLayer(g_hidden, g_hidden, dropout)
        self.lstm = nn.LSTM(g_hidden, lstm_hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden // 2, 1),
        )

    def forward(self, x_seq: torch.Tensor, nbr_idx_seq: torch.Tensor, nbr_w_seq: torch.Tensor) -> torch.Tensor:
        hs = []
        for u in range(x_seq.shape[0]):
            h = self.gcn1(x_seq[u], nbr_idx_seq[u], nbr_w_seq[u])
            h = self.gcn2(h, nbr_idx_seq[u], nbr_w_seq[u])
            hs.append(h)
        z = torch.stack(hs, dim=1)  # [N,L,H]
        out, _ = self.lstm(z)
        return self.head(out[:, -1, :]).squeeze(-1)


def r2_np(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) < 2:
        return np.nan
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    return np.nan if sst <= 1e-12 else float(1 - sse / sst)


def rank_ic_by_day(df: pd.DataFrame):
    vals = []
    for _, g in df.groupby("date"):
        if len(g) < 10 or g["y_true"].nunique() < 3 or g["y_pred"].nunique() < 3:
            continue
        rho = spearmanr(g["y_true"].values, g["y_pred"].values).correlation
        if np.isfinite(rho):
            vals.append(float(rho))
    return float(np.mean(vals)) if vals else np.nan


def ic_by_day(df: pd.DataFrame):
    vals = []
    for _, g in df.groupby("date"):
        if len(g) < 10:
            continue
        c = np.corrcoef(g["y_true"].values, g["y_pred"].values)[0, 1]
        if np.isfinite(c):
            vals.append(float(c))
    return float(np.mean(vals)) if vals else np.nan


def topk_metrics(df: pd.DataFrame, ks=(1, 3, 5, 10)):
    rows = []
    for k in ks:
        daily = []
        for d, g in df.groupby("date"):
            top = g.sort_values("y_pred", ascending=False).head(k)
            if len(top):
                daily.append({"date": d, "ret": float(top["y_true"].mean()), "win": float((top["y_true"] > 0).mean())})
        if daily:
            tmp = pd.DataFrame(daily)
            rows.append({
                "k": k,
                "n_days": int(len(tmp)),
                "mean_daily_topk_return": float(tmp["ret"].mean()),
                "std_daily_topk_return": float(tmp["ret"].std(ddof=1)),
                "topk_win_rate": float(tmp["win"].mean()),
            })
    return pd.DataFrame(rows)


def newey_west_se(x, lag=2):
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
        var += 2 * (1 - l / (lag + 1)) * gamma
    return math.sqrt(max(var, 0) / T)


def dm_vs_baseline(pred_df: pd.DataFrame, base_csv: str, lag: int = 2) -> Dict[str, float]:
    base = pd.read_csv(base_csv)
    rename = {}
    if "pred" in base.columns and "y_pred" not in base.columns:
        rename["pred"] = "y_pred"
    if "prediction" in base.columns and "y_pred" not in base.columns:
        rename["prediction"] = "y_pred"
    if "y" in base.columns and "y_true" not in base.columns:
        rename["y"] = "y_true"
    if "ts_code" in base.columns and "stock" not in base.columns:
        rename["ts_code"] = "stock"
    base = base.rename(columns=rename)
    if not {"date", "stock", "y_pred"}.issubset(base.columns):
        return {"dm_mean_d_vs_base": np.nan, "dm_stat_vs_base": np.nan, "dm_p_one_sided_vs_base": np.nan}
    pred = pred_df.copy()
    pred["date"] = pred["date"].astype(str)
    base["date"] = base["date"].astype(str)
    pred["stock"] = pred["stock"].astype(str)
    base["stock"] = base["stock"].astype(str)
    merged = pred[["date", "stock", "y_true", "y_pred"]].merge(
        base[["date", "stock", "y_pred"]], on=["date", "stock"], how="inner", suffixes=("", "_base")
    )
    if len(merged) == 0:
        return {"dm_mean_d_vs_base": np.nan, "dm_stat_vs_base": np.nan, "dm_p_one_sided_vs_base": np.nan}
    ds = []
    for _, g in merged.groupby("date"):
        e_model = (g["y_true"].values - g["y_pred"].values) ** 2
        e_base = (g["y_true"].values - g["y_pred_base"].values) ** 2
        ds.append(float(np.mean(e_base - e_model)))
    d = np.asarray(ds)
    mean_d = float(np.mean(d))
    se = newey_west_se(d, lag)
    dm = mean_d / se if np.isfinite(se) and se > 0 else np.nan
    p = float(1 - norm.cdf(dm)) if np.isfinite(dm) else np.nan
    return {"dm_mean_d_vs_base": mean_d, "dm_stat_vs_base": float(dm), "dm_p_one_sided_vs_base": p}


def summarize(pred_df: pd.DataFrame, baseline_csv: Optional[str] = None, dm_lag: int = 2):
    y_true, y_pred = pred_df["y_true"].values, pred_df["y_pred"].values
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    pred_df = pred_df.copy()
    pred_df["month"] = pd.to_datetime(pred_df["date"]).dt.to_period("M").astype(str)
    monthly = []
    for m, g in pred_df.groupby("month"):
        monthly.append({
            "month": m,
            "model": "gcn_lstm",
            "n_obs": int(len(g)),
            "r2": r2_np(g["y_true"], g["y_pred"]),
            "mse": float(np.mean((g["y_true"] - g["y_pred"]) ** 2)),
            "mae": float(np.mean(np.abs(g["y_true"] - g["y_pred"]))),
            "rank_ic": rank_ic_by_day(g),
        })
    monthly_df = pd.DataFrame(monthly)
    weighted_monthly_r2 = np.nan
    if len(monthly_df):
        w = monthly_df["n_obs"].values / monthly_df["n_obs"].sum()
        weighted_monthly_r2 = float(np.nansum(w * monthly_df["r2"].values))
    summary = {
        "model": "gcn_lstm",
        "n_obs": int(len(pred_df)),
        "oos_r2_all": r2_np(y_true, y_pred),
        "weighted_monthly_r2": weighted_monthly_r2,
        "mse_all": mse,
        "mae_all": mae,
        "rmse_all": float(np.sqrt(mse)),
        "ic_daily_mean": ic_by_day(pred_df),
        "rank_ic_daily_mean": rank_ic_by_day(pred_df),
    }
    if baseline_csv:
        summary.update(dm_vs_baseline(pred_df, baseline_csv, lag=dm_lag))
    return summary, monthly_df, topk_metrics(pred_df)


def evaluate(model, X, y, valid, dates, stocks, nbr_idx, nbr_w, node_valid, target_days, lookback, mu, sd, device, min_nodes, amp=False, save_pred=False):
    model.eval()
    loss_list, rows = [], []
    use_amp = amp and device.type == "cuda"
    with torch.no_grad():
        for t in map(int, target_days):
            m_np = valid_nodes_for_t(valid, node_valid, t, lookback, min_nodes)
            if m_np is None:
                continue
            x_t, ni_t, nw_t = prepare_day_tensors(X, nbr_idx, nbr_w, t, lookback, mu, sd, device)
            yy_np = np.asarray(y[t, :], dtype=np.float32)
            yy = torch.as_tensor(yy_np, dtype=torch.float32, device=device)
            m = torch.as_tensor(m_np, dtype=torch.bool, device=device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x_t, ni_t, nw_t)
                loss = torch.mean((pred[m] - yy[m]) ** 2)
            loss_list.append(float(loss.detach().cpu()))
            if save_pred:
                idx = np.where(m_np)[0]
                rows.append(pd.DataFrame({
                    "date": str(pd.Timestamp(dates[t]).date()),
                    "t_idx": t,
                    "stock": stocks[idx],
                    "n_idx": idx,
                    "y_true": yy_np[idx],
                    "y_pred": pred.detach().cpu().numpy()[idx].astype(np.float32),
                    "model": "gcn_lstm",
                }))
    pred_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return float(np.mean(loss_list)) if loss_list else np.nan, pred_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--graph_cache", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--train_start", default="2022-01-01")
    ap.add_argument("--train_end", default="2024-12-31")
    ap.add_argument("--test_start", default="2025-01-01")
    ap.add_argument("--test_end", default="2025-12-31")
    ap.add_argument("--valid_ratio", type=float, default=0.20)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--g_hidden", type=int, default=64)
    ap.add_argument("--lstm_hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--min_nodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--zero_pred_csv", default=None, help="Optional baseline prediction CSV for DM test.")
    ap.add_argument("--dm_lag", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, valid, dates, stocks, nbr_idx, nbr_w, node_valid = load_data(args.data_dir, args.graph_cache)
    T, N, F = X.shape
    print(f"Loaded X={X.shape}, y={y.shape}, nbr_idx={nbr_idx.shape}, device={args.device}")
    print(f"dates={dates[0]}..{dates[-1]}, stocks={len(stocks)}, F={F}")

    train_t, val_t, test_t = make_target_days(dates, args.lookback, args.train_start, args.train_end, args.test_start, args.test_end, args.valid_ratio)
    print(f"train days={len(train_t)}: {dates[train_t[0]]}..{dates[train_t[-1]]}")
    print(f"val days={len(val_t)}: {dates[val_t[0]]}..{dates[val_t[-1]]}")
    print(f"test days={len(test_t)}: {dates[test_t[0]]}..{dates[test_t[-1]]}")

    mu, sd = calc_feature_scaler(X, valid, train_t, args.lookback)
    np.save(out_dir / "feature_mu.npy", mu)
    np.save(out_dir / "feature_sd.npy", sd)

    device = torch.device(args.device)
    model = GCNLSTM(F, args.g_hidden, args.lstm_hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val, best_state, bad = np.inf, None, 0
    hist = []

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        days = list(map(int, train_t))
        random.shuffle(days)
        losses, used = [], 0
        for t in days:
            m_np = valid_nodes_for_t(valid, node_valid, t, args.lookback, args.min_nodes)
            if m_np is None:
                continue
            x_t, ni_t, nw_t = prepare_day_tensors(X, nbr_idx, nbr_w, t, args.lookback, mu, sd, device)
            yy = torch.as_tensor(np.asarray(y[t, :], dtype=np.float32), dtype=torch.float32, device=device)
            m = torch.as_tensor(m_np, dtype=torch.bool, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                pred = model(x_t, ni_t, nw_t)
                loss = torch.mean((pred[m] - yy[m]) ** 2)
            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            used += 1
        train_loss = float(np.mean(losses)) if losses else np.nan
        val_loss, _ = evaluate(model, X, y, valid, dates, stocks, nbr_idx, nbr_w, node_valid, val_t, args.lookback, mu, sd, device, args.min_nodes, amp=args.amp, save_pred=False)
        elapsed = time.time() - t0
        print(f"epoch {ep:02d}/{args.epochs} train_mse={train_loss:.6f} val_mse={val_loss:.6f} used_days={used} time={elapsed:.1f}s")
        hist.append({"epoch": ep, "train_mse": train_loss, "val_mse": val_loss, "used_days": used, "elapsed_sec": elapsed})
        if np.isfinite(val_loss) and val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, out_dir / "best_gcn_lstm.pt")
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {ep}. best_val_mse={best_val:.6f}")
                break

    pd.DataFrame(hist).to_csv(out_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, pred_df = evaluate(model, X, y, valid, dates, stocks, nbr_idx, nbr_w, node_valid, test_t, args.lookback, mu, sd, device, args.min_nodes, amp=args.amp, save_pred=True)
    pred_df.to_csv(out_dir / "pred_gcn_lstm.csv", index=False, encoding="utf-8-sig")

    summary, monthly_df, topk_df = summarize(pred_df, baseline_csv=args.zero_pred_csv, dm_lag=args.dm_lag)
    summary.update({
        "test_mse_daily_mean": test_loss,
        "best_val_mse": float(best_val),
        "epochs_run": len(hist),
        "device": str(device),
        "amp": bool(args.amp and device.type == "cuda"),
    })
    pd.DataFrame([summary]).to_csv(out_dir / "summary_gcn_lstm.csv", index=False, encoding="utf-8-sig")
    monthly_df.to_csv(out_dir / "monthly_gcn_lstm.csv", index=False, encoding="utf-8-sig")
    topk_df.to_csv(out_dir / "topk_gcn_lstm.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "config_gcn_lstm.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Predictions: {out_dir / 'pred_gcn_lstm.csv'}")
    print(f"Summary: {out_dir / 'summary_gcn_lstm.csv'}")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
