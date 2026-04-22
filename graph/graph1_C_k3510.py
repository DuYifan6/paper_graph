# -*- coding: utf-8 -*-
"""
Graph1 RealGraph | Strong-graph tuned C version for K=3/5/10

Purpose
- Keep the ORIGINAL strong graph settings as the default.
- Add validation split + best checkpoint + early stopping.
- Make Laplacian and ranking loss switchable so the user can run:
  A) strong graph + pure MSE + val selection
  B) strong graph + no Laplacian + val selection
  C) strong graph + no Laplacian + light ranking loss + val selection
- Export train-quarter and test-month compare tables vs fixed ZeroGraph.
- Plot R^2 comparison figures.

How to use
- Just run the script.
"""

import os
import glob
import math
import json
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

# =======================
# 0) PROFILE / 配置
# =======================
PROFILE = "C_K3510"   # fixed tuned C version for K=3/5/10
SEED = 0

CACHE_NPZ = r"D:\PythonProject\LASSO_FINAL\common_preprocess_cache\common_tensor_cache.npz"
META_PKL  = r"D:\PythonProject\LASSO_FINAL\common_preprocess_cache\common_tensor_meta.pkl"

EXP_NAME = f"Graph1_{PROFILE}"

# 固定 ZeroGraph 结果（用于 compare 和 R² 图）
ZERO_TRAIN_DETAIL = r"D:\PythonProject\LASSO_FINAL\graph_调整\fixed_zerograph_valselect\results\predictions_ZeroGraph_train_detail.csv"
ZERO_TEST_DETAIL  = r"D:\PythonProject\LASSO_FINAL\graph_调整\fixed_zerograph_valselect\results\predictions_ZeroGraph_test_detail.csv"

# ===== 图一：5min 数据目录 =====
MINUTE5_DIR = r"D:\Lasso\min5_aligned"
MINUTE5_SUFFIX_CANDIDATES = ["_min.csv", "_5m.csv", "_5分钟.csv", ".csv"]
TIME_COL = "trade_time"
CLOSE_COL = "close"

# bars/returns per day
BARS_PER_DAY = 49
RET_PER_DAY  = BARS_PER_DAY - 1

# ===== 图一：原始强图参数（默认保留） =====
ROLL_W = 20
TOPK = 40
GRAPH_SHIFT_TRADINGDAY = 0

ABS_CORR = True
MARKET_REMOVE = True
CLIP_RET = 0.08

EDGE_TAU = 0.10
EDGE_POWER = 1.5

REQUIRE_FULL_WINDOW = True
MIN_DAYS_IN_WINDOW = 16
GPU_BLOCK = 256

# ===== 模型参数 =====
LOOKBACK = 20
HIDDEN = 128
NUM_LAYERS = 1
DROPOUT = 0.1
LR = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
GRAD_CLIP = 1.0
GATE_HIDDEN = 128

MIN_VALID_NODES_PER_DAY = 50

USE_SYM_NORM_IN_GRAPHMIX = True
USE_NORMALIZED_LAPLACIAN = True

# ===== validation / 选模 =====
VALID_RATIO = 0.20
EARLY_STOP_PATIENCE = 4
SELECT_KS = [1, 3, 5, 10]
SELECT_WEIGHTS = {1: 0.05, 3: 0.45, 5: 0.30, 10: 0.20}

# ===== ranking loss defaults =====
RANK_TOP_FRAC = 0.15
RANK_MIN_TOP = 5
RANK_MAX_TOP = 25

DM_LAG = 2
DM_ALTERNATIVE = "greater"

# ===== 按 profile 设置 ablation =====
if PROFILE == "C_K3510":
    # strong graph + no Laplacian + ranking loss + validation selection oriented to K=3/5/10
    LAPLACIAN_LAM = 0.0
    USE_RANK_LOSS = True
    RANK_LOSS_LAM = 0.05
else:
    raise ValueError("PROFILE must be: C_K3510")

# ===== 输出目录 =====
OUT_DIR = rf"D:\PythonProject\LASSO_FINAL\graph_调整\graph1_stronggraph"
CACHE_DIR  = os.path.join(OUT_DIR, "cache")
GRAPH_DIR  = os.path.join(OUT_DIR, "graphs")
RESULT_DIR = os.path.join(OUT_DIR, "results")
PLOT_DIR   = os.path.join(OUT_DIR, "plots")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

RET_MEMMAP_PATH = None
RET_VALID_PATH  = None
GRAPH_CACHE = os.path.join(
    GRAPH_DIR,
    f"roll{ROLL_W}_abs{int(ABS_CORR)}_mkt{int(MARKET_REMOVE)}_topk{TOPK}_shift{GRAPH_SHIFT_TRADINGDAY}"
    f"_tau{EDGE_TAU}_pow{EDGE_POWER}_full{int(REQUIRE_FULL_WINDOW)}.npz"
)


# =======================
# 1) 工具
# =======================
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


try:
    from scipy.stats import t as student_t
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _newey_west_var(d: np.ndarray, lag: int) -> float:
    d = np.asarray(d, dtype=np.float64)
    T = d.size
    if T < 2:
        return np.nan
    lag = int(max(0, min(lag, T - 1)))

    x = d - d.mean()
    gamma0 = np.dot(x, x) / T
    hac = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma_k = np.dot(x[k:], x[:-k]) / T
        hac += 2.0 * w * gamma_k
    var_bar = hac / T
    return float(max(var_bar, 1e-12))


def dm_test(d: np.ndarray, lag: int = 0, alternative: str = "greater") -> Dict[str, Any]:
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    T = int(d.size)
    if T < 5:
        return {"T": T, "dm": np.nan, "p_one": np.nan, "p_two": np.nan, "mean_d": float(np.nan), "lag": int(lag)}

    mean_d = float(d.mean())
    var_bar = _newey_west_var(d, lag=lag)
    dm = mean_d / np.sqrt(var_bar)
    df = max(T - 1, 1)

    if _HAS_SCIPY:
        cdf = float(student_t.cdf(dm, df=df))
        if alternative == "greater":
            p_one = float(1.0 - cdf)
        elif alternative == "less":
            p_one = float(cdf)
        else:
            p_one = float(np.nan)
        p_two = float(2.0 * min(cdf, 1.0 - cdf))
    else:
        cdf = 0.5 * (1.0 + math.erf(dm / math.sqrt(2.0)))
        if alternative == "greater":
            p_one = float(1.0 - cdf)
        elif alternative == "less":
            p_one = float(cdf)
        else:
            p_one = float(np.nan)
        p_two = float(2.0 * min(cdf, 1.0 - cdf))

    return {"T": T, "dm": float(dm), "p_one": p_one, "p_two": p_two, "mean_d": mean_d, "lag": int(lag)}


def build_train_val_split(train_start_idx: int, train_end_idx: int, valid_ratio: float = 0.2):
    total_days = train_end_idx - train_start_idx + 1
    val_days = max(20, int(round(total_days * valid_ratio)))
    val_start_idx = max(train_start_idx + LOOKBACK, train_end_idx - val_days + 1)
    fit_end_idx = val_start_idx - 1
    if fit_end_idx < train_start_idx + LOOKBACK:
        val_start_idx = train_end_idx + 1
        fit_end_idx = train_end_idx
    return fit_end_idx, val_start_idx


def daily_topk_mean_return(pred_np: np.ndarray, y_np: np.ndarray, ks=(1, 3, 5, 10)):
    if len(pred_np) == 0:
        return {k: np.nan for k in ks}
    order = np.argsort(-pred_np)
    y_sorted = y_np[order]
    out = {}
    for k in ks:
        kk = min(k, len(y_sorted))
        out[k] = float(np.mean(y_sorted[:kk])) if kk > 0 else np.nan
    return out


def compute_selection_score_from_detail(detail_df: pd.DataFrame):
    if detail_df is None or detail_df.empty:
        return {
            "select_score": -1e18,
            "ret_at_1": np.nan,
            "ret_at_3": np.nan,
            "ret_at_5": np.nan,
            "ret_at_10": np.nan,
        }

    rows = []
    df = detail_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for d, g in df.groupby("date", sort=True):
        pred_np = g["y_pred"].to_numpy(dtype=float)
        true_np = g["y_true"].to_numpy(dtype=float)
        topk_map = daily_topk_mean_return(pred_np, true_np, ks=SELECT_KS)
        row = {"date": d}
        for k in SELECT_KS:
            row[f"ret_at_{k}"] = topk_map[k]
        rows.append(row)

    daily_df = pd.DataFrame(rows)
    agg = {f"ret_at_{k}": float(daily_df[f"ret_at_{k}"].mean()) for k in SELECT_KS}
    score = 0.0
    for k in SELECT_KS:
        score += SELECT_WEIGHTS[k] * agg[f"ret_at_{k}"]
    agg["select_score"] = float(score)
    return agg


def ranking_loss_top_vs_rest(pred: torch.Tensor, y: torch.Tensor):
    n = pred.numel()
    if n < 10:
        return pred.new_tensor(0.0)

    order = torch.argsort(y, descending=True)
    top_n = max(RANK_MIN_TOP, int(round(n * RANK_TOP_FRAC)))
    top_n = min(top_n, RANK_MAX_TOP, n - 1)
    top_idx = order[:top_n]
    rest_idx = order[top_n:]
    if rest_idx.numel() == 0:
        return pred.new_tensor(0.0)

    p_top = pred[top_idx].view(-1, 1)
    p_rest = pred[rest_idx].view(1, -1)
    return F.softplus(-(p_top - p_rest)).mean()


# =======================
# 2) 图一：5min 文件读取与收益构建
# =======================
def find_5m_file(code: str) -> Optional[str]:
    for suf in MINUTE5_SUFFIX_CANDIDATES:
        p = os.path.join(MINUTE5_DIR, f"{code}{suf}")
        if os.path.exists(p):
            return p
    cands = glob.glob(os.path.join(MINUTE5_DIR, f"{code}*.csv"))
    if cands:
        cands2 = [x for x in cands if "5" in os.path.basename(x)]
        return cands2[0] if cands2 else cands[0]
    return None


def compute_logret_from_close(close_arr: np.ndarray) -> Optional[np.ndarray]:
    close_arr = close_arr.astype(np.float32)
    close_arr = close_arr[np.isfinite(close_arr)]
    if close_arr.size < BARS_PER_DAY:
        return None
    close_arr = close_arr[-BARS_PER_DAY:]
    if np.any(close_arr <= 0):
        return None
    logc = np.log(close_arr + 1e-12)
    ret = np.diff(logc).astype(np.float32)
    if ret.size != RET_PER_DAY or (not np.all(np.isfinite(ret))):
        return None
    if CLIP_RET and CLIP_RET > 0:
        ret = np.clip(ret, -float(CLIP_RET), float(CLIP_RET))
    return ret


def build_or_load_returns_memmap(dates: List[pd.Timestamp], codes: List[str]):
    global RET_MEMMAP_PATH, RET_VALID_PATH

    if not os.path.isdir(MINUTE5_DIR):
        raise FileNotFoundError(f"找不到 MINUTE5_DIR={MINUTE5_DIR}")

    T = len(dates)
    N = len(codes)
    M = RET_PER_DAY

    RET_MEMMAP_PATH = os.path.join(CACHE_DIR, f"ret5m_logret_T{T}_N{N}_M{M}.dat")
    RET_VALID_PATH  = os.path.join(CACHE_DIR, f"ret5m_valid_T{T}_N{N}.npy")

    if os.path.exists(RET_MEMMAP_PATH) and os.path.exists(RET_VALID_PATH):
        ret = np.memmap(RET_MEMMAP_PATH, dtype="float32", mode="r", shape=(T, N, M))
        valid = np.load(RET_VALID_PATH).astype(np.uint8)
        print(f"[INFO] 使用 returns 缓存: {RET_MEMMAP_PATH}")
        return ret, valid

    print("[INFO] 读取 5min 并构建 logret memmap（首次运行较慢，后续直接复用）")
    date_to_t = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
    date_set = set(date_to_t.keys())
    code_to_n = {c: i for i, c in enumerate(codes)}

    ret = np.memmap(RET_MEMMAP_PATH, dtype="float32", mode="w+", shape=(T, N, M))
    ret[:] = 0.0
    valid = np.zeros((T, N), dtype=np.uint8)

    missing_files = 0
    for code in tqdm(codes, desc="5min->logret"):
        fpath = find_5m_file(code)
        if fpath is None:
            missing_files += 1
            continue

        ni = code_to_n[code]
        try:
            df = pd.read_csv(fpath, usecols=[TIME_COL, CLOSE_COL], low_memory=False)
        except Exception:
            missing_files += 1
            continue

        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.dropna(subset=[TIME_COL])
        if df.empty:
            continue

        df["date"] = df[TIME_COL].dt.normalize()
        df = df[df["date"].isin(date_set)].copy()
        if df.empty:
            continue

        df[CLOSE_COL] = pd.to_numeric(df[CLOSE_COL], errors="coerce").astype(np.float32)
        df = df.dropna(subset=[CLOSE_COL])
        if df.empty:
            continue

        df = df.sort_values(TIME_COL)
        for d, sub in df.groupby("date", sort=False):
            ti = date_to_t.get(pd.Timestamp(d).normalize(), None)
            if ti is None:
                continue

            sub = sub.tail(BARS_PER_DAY)
            r = compute_logret_from_close(sub[CLOSE_COL].to_numpy())
            if r is None:
                continue

            ret[ti, ni, :] = r
            valid[ti, ni] = 1

    ret.flush()
    np.save(RET_VALID_PATH, valid)
    print(f"[INFO] returns 构建完成 | missing_files={missing_files} | 保存: {RET_MEMMAP_PATH}")
    return np.memmap(RET_MEMMAP_PATH, dtype="float32", mode="r", shape=(T, N, M)), valid


# =======================
# 3) 图一：Corr-TopK
# =======================
@torch.no_grad()
def corr_topk_gpu_block(X: np.ndarray, topk: int, device: torch.device, block: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    Xt = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    N, L = Xt.shape
    denom = float(max(L - 1, 1))

    idx_out = torch.empty((N, topk), device="cpu", dtype=torch.int64)
    val_out = torch.empty((N, topk), device="cpu", dtype=torch.float32)

    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        S = (Xt[i0:i1] @ Xt.T) / denom
        S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        if ABS_CORR:
            S = torch.abs(S)
        for r in range(i1 - i0):
            S[r, i0 + r] = -1e9
        v, ix = torch.topk(S, k=topk, dim=1, largest=True, sorted=True)
        idx_out[i0:i1] = ix.detach().cpu()
        val_out[i0:i1] = v.detach().cpu()
    return idx_out.numpy(), val_out.numpy()


def corr_topk_cpu_exact(X: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.ascontiguousarray(X, dtype=np.float32)
    N, L = X.shape
    denom = np.float32(max(L - 1, 1))

    S = (X @ X.T) / denom
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if ABS_CORR:
        S = np.abs(S)
    np.fill_diagonal(S, -1e9)

    part = np.argpartition(-S, topk - 1, axis=1)[:, :topk]
    part_val = np.take_along_axis(S, part, axis=1)
    order = np.argsort(-part_val, axis=1)

    idx = np.take_along_axis(part, order, axis=1).astype(np.int64)
    val = np.take_along_axis(part_val, order, axis=1).astype(np.float32)
    return idx, val


def symmetrize_union_and_retopk(nbr_idx: np.ndarray, nbr_w: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    N = nbr_idx.shape[0]
    rows = np.repeat(np.arange(N, dtype=np.int64), topk)
    cols = nbr_idx.reshape(-1).astype(np.int64)
    vals = nbr_w.reshape(-1).astype(np.float32)

    m = np.isfinite(vals) & (vals > 0) & (cols != rows)
    rows, cols, vals = rows[m], cols[m], vals[m]

    rows2 = np.concatenate([rows, cols])
    cols2 = np.concatenate([cols, rows])
    vals2 = np.concatenate([vals, vals])

    order = np.lexsort((-vals2, rows2))
    rows2, cols2, vals2 = rows2[order], cols2[order], vals2[order]

    out_idx = np.tile(np.arange(N, dtype=np.uint16)[:, None], (1, topk))
    out_w = np.zeros((N, topk), dtype=np.float32)

    ptr = 0
    for i in range(N):
        while ptr < rows2.size and rows2[ptr] < i:
            ptr += 1
        j = ptr
        seen = set()
        kfill = 0
        while j < rows2.size and rows2[j] == i and kfill < topk:
            nb = int(cols2[j])
            if nb != i and nb not in seen:
                out_idx[i, kfill] = np.uint16(nb)
                out_w[i, kfill] = float(vals2[j])
                seen.add(nb)
                kfill += 1
            j += 1
        ptr = j
    return out_idx, out_w


def build_rolling_corr_graph(dates: List[pd.Timestamp], codes: List[str], ret_mmap: np.memmap, valid: np.ndarray):
    if os.path.exists(GRAPH_CACHE):
        pack = np.load(GRAPH_CACHE, allow_pickle=True)
        g_dates = pack["dates"].astype("datetime64[ns]")
        nbr_idx = pack["nbr_idx"]
        nbr_w = pack["nbr_w"].astype(np.float32)
        node_valid = pack["node_valid"].astype(np.uint8)
        meta = json.loads(str(pack["meta"].item()))
        print(f"[INFO] 使用图缓存: {GRAPH_CACHE} | T={len(g_dates)} N={nbr_idx.shape[1]} K={nbr_idx.shape[2]}")
        print("[INFO] meta:", meta)
        return g_dates, nbr_idx, nbr_w, node_valid

    T = len(dates)
    N = len(codes)
    topk_eff = min(TOPK, max(1, N - 1))
    L = ROLL_W * RET_PER_DAY

    nbr_idx_all = np.tile(np.arange(N, dtype=np.uint16)[None, :, None], (T, 1, topk_eff))
    nbr_w_all = np.zeros((T, N, topk_eff), dtype=np.float32)
    node_valid = np.zeros((T, N), dtype=np.uint8)

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    if use_gpu:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print(f"[INFO] 构图：rolling corr | use_gpu={use_gpu} device={device}")
    print(f"[INFO] T={T} N={N} window={ROLL_W} L={L} TOPK={topk_eff} market_remove={MARKET_REMOVE} require_full={REQUIRE_FULL_WINDOW}")

    dates_np = pd.to_datetime(pd.Series(dates)).to_numpy(dtype="datetime64[ns]")

    for t in tqdm(range(T), desc="build roll20 corr graphs"):
        if t < ROLL_W:
            continue

        win_ret = np.array(ret_mmap[t - ROLL_W:t, :, :], dtype=np.float32)
        win_val = valid[t - ROLL_W:t, :].astype(bool)

        if REQUIRE_FULL_WINDOW:
            ok = win_val.all(axis=0)
        else:
            ok = (win_val.sum(axis=0) >= MIN_DAYS_IN_WINDOW)

        node_valid[t] = ok.astype(np.uint8)
        if ok.sum() < max(200, topk_eff + 5):
            continue

        X = win_ret.transpose(1, 0, 2).reshape(N, L)
        if MARKET_REMOVE:
            mkt = X[ok].mean(axis=0, keepdims=True)
            X = X - mkt
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-6
        X = (X - mu) / sd
        X[~ok] = 0.0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if use_gpu:
            idx, val = corr_topk_gpu_block(X, topk_eff, device=device, block=GPU_BLOCK)
        else:
            idx, val = corr_topk_cpu_exact(X, topk_eff)

        val = np.clip(val, 0.0, 1.0)
        if EDGE_TAU and EDGE_TAU > 0:
            val = np.maximum(val - float(EDGE_TAU), 0.0)
        if EDGE_POWER and EDGE_POWER != 1.0:
            val = np.power(val, float(EDGE_POWER))

        bad = ~ok
        if bad.any():
            idx[bad] = np.arange(N, dtype=np.int64)[bad][:, None]
            val[bad] = 0.0

        idx_sym, val_sym = symmetrize_union_and_retopk(idx, val, topk_eff)
        nbr_idx_all[t] = idx_sym.astype(np.uint16)
        nbr_w_all[t]   = val_sym.astype(np.float32)

    meta = dict(
        graph_type="rolling_abs_corr_from_5m_logret",
        roll_window_days=ROLL_W,
        ret_per_day=RET_PER_DAY,
        topk=topk_eff,
        abs_corr=bool(ABS_CORR),
        market_remove=bool(MARKET_REMOVE),
        clip_ret=float(CLIP_RET),
        edge_tau=float(EDGE_TAU),
        edge_power=float(EDGE_POWER),
        undirected_union=True,
        no_self_loop=True,
        note="Graph(day t) uses window [t-ROLL_W, t-1]. Training uses graph(t-1) to predict y(t)."
    )

    np.savez_compressed(
        GRAPH_CACHE,
        dates=dates_np,
        nbr_idx=nbr_idx_all.astype(np.uint16),
        nbr_w=nbr_w_all.astype(np.float32),
        node_valid=node_valid.astype(np.uint8),
        meta=json.dumps(meta, ensure_ascii=False)
    )
    print(f"[INFO] 已保存图缓存 -> {GRAPH_CACHE}")
    return dates_np, nbr_idx_all, nbr_w_all, node_valid


# =======================
# 4) 图工具
# =======================
def knn_sym_norm_weights(nbr_idx: torch.Tensor, nbr_w_raw: torch.Tensor, eps: float = 1e-6):
    w = torch.nan_to_num(nbr_w_raw, nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.clamp(w, min=0.0)

    deg = w.sum(dim=1) + eps
    deg_i = deg.view(-1, 1)
    deg_j = deg.index_select(0, nbr_idx.reshape(-1)).view_as(w) + eps

    w_norm = w / torch.sqrt(deg_i * deg_j)
    w_norm = torch.nan_to_num(w_norm, nan=0.0, posinf=0.0, neginf=0.0)
    w_norm = torch.clamp(w_norm, min=0.0)
    return w_norm, deg


def laplacian_loss_knn_unnorm(pred: torch.Tensor, nbr_idx: torch.Tensor, nbr_w_raw: torch.Tensor):
    w = torch.nan_to_num(nbr_w_raw, nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.clamp(w, min=0.0)
    nbr_pred = pred.index_select(0, nbr_idx.reshape(-1)).view_as(w)
    diff = pred.view(-1, 1) - nbr_pred
    return (w * diff * diff).mean()


def laplacian_loss_knn_normalized(pred: torch.Tensor, nbr_idx: torch.Tensor, nbr_w_raw: torch.Tensor, deg: torch.Tensor, eps: float = 1e-6):
    w = torch.nan_to_num(nbr_w_raw, nan=0.0, posinf=0.0, neginf=0.0)
    w = torch.clamp(w, min=0.0)
    y_norm = pred / torch.sqrt(deg + eps)
    y_nb = y_norm.index_select(0, nbr_idx.reshape(-1)).view_as(w)
    diff = y_norm.view(-1, 1) - y_nb
    return (w * diff * diff).mean()


# =======================
# 5) 模型
# =======================
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
        if node_valid_mask is not None:
            Z_send = Z * node_valid_mask.view(N, 1)
        else:
            Z_send = Z

        nbr_w = torch.nan_to_num(nbr_w, nan=0.0, posinf=0.0, neginf=0.0)
        nbr_w = torch.clamp(nbr_w, min=0.0)

        Z_nb_raw = Z_send.index_select(0, nbr_idx.reshape(-1)).view(N, K, H)
        w = nbr_w.view(N, K, 1)
        num = (w * Z_nb_raw).sum(dim=1)
        den = (nbr_w.sum(dim=1, keepdim=True) + 1e-6)
        Z_nb = num / den

        cat = torch.cat([Z, Z_nb], dim=1)
        g = torch.sigmoid(self.lin_gate2(F.relu(self.lin_gate(cat))))
        delta = self.dropout(F.relu(self.lin_msg(cat)))
        return Z + g * delta


class GraphAugLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 1, dropout: float = 0.1, gate_hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.graph = GraphMixKNN(hidden=hidden, gate_hidden=gate_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, X, nbr_idx=None, nbr_w=None, node_valid_mask=None):
        out, _ = self.lstm(X)
        z = self.dropout(out[:, -1, :])
        z = self.graph(z, nbr_idx=nbr_idx, nbr_w=nbr_w, node_valid_mask=node_valid_mask)
        return self.head(z).squeeze(-1)


# =======================
# 6) 训练 / 评估
# =======================
def train_model_on_range(
    X_np, y_np, mask_np, dates, ts_codes_order,
    graph_dates, graph_nbr_idx, graph_nbr_w,
    in_dim: int, device: torch.device,
    train_start_idx: int, train_end_idx: int,
    seed: int = 0
):
    set_seed(seed)

    model = GraphAugLSTM(
        in_dim=in_dim,
        hidden=HIDDEN,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        gate_hidden=GATE_HIDDEN
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    m = torch.from_numpy(mask_np).to(device)

    fit_end_idx, val_start_idx = build_train_val_split(
        train_start_idx=train_start_idx,
        train_end_idx=train_end_idx,
        valid_ratio=VALID_RATIO
    )
    use_validation = val_start_idx <= train_end_idx

    def graph_pos_for_pred_date(d_pred: pd.Timestamp):
        d = np.datetime64(d_pred.normalize().to_datetime64())
        pos = np.searchsorted(graph_dates, d) - int(GRAPH_SHIFT_TRADINGDAY)
        if pos < 0:
            return None
        return int(pos)

    def get_graph_for_t(t_idx: int):
        pos = graph_pos_for_pred_date(pd.Timestamp(dates[t_idx]))
        if pos is None or pos >= len(graph_dates):
            return None, None, None, None

        nbr_i = torch.from_numpy(graph_nbr_idx[pos].astype(np.int64)).to(device)
        nbr_w_raw = torch.from_numpy(graph_nbr_w[pos].astype(np.float32)).to(device)
        nbr_w_raw = torch.nan_to_num(nbr_w_raw, nan=0.0, posinf=0.0, neginf=0.0)
        nbr_w_raw = torch.clamp(nbr_w_raw, min=0.0)

        if USE_SYM_NORM_IN_GRAPHMIX:
            nbr_w_msg, deg_raw = knn_sym_norm_weights(nbr_i, nbr_w_raw)
        else:
            deg_raw = nbr_w_raw.sum(dim=1) + 1e-6
            nbr_w_msg = nbr_w_raw

        return nbr_i, nbr_w_msg, nbr_w_raw, deg_raw

    def evaluate_for_selection(eval_start_idx: int, eval_end_idx: int):
        model.eval()
        detail_rows = []
        with torch.no_grad():
            loop_start = max(LOOKBACK - 1, eval_start_idx)
            for t in range(loop_start, eval_end_idx + 1):
                mt = m[t]
                if mt.sum().item() < MIN_VALID_NODES_PER_DAY:
                    continue

                seq = X[t - LOOKBACK + 1:t + 1].permute(1, 0, 2).contiguous()
                yt = y[t]
                node_valid = mt.float()

                nbr_i, nbr_w_msg, _, _ = get_graph_for_t(t)
                pred = model(seq, nbr_idx=nbr_i, nbr_w=nbr_w_msg, node_valid_mask=node_valid)
                if not torch.isfinite(pred[mt]).all():
                    continue

                pred_np = pred[mt].detach().cpu().numpy()
                true_np = yt[mt].detach().cpu().numpy()
                node_indices = np.where(mt.cpu().numpy())[0]
                current_date = pd.Timestamp(dates[t]).normalize()

                for node_idx, pred_val, true_val in zip(node_indices, pred_np, true_np):
                    detail_rows.append({
                        "date": current_date,
                        "stock": ts_codes_order[node_idx],
                        "y_pred": float(pred_val),
                        "y_true": float(true_val),
                    })

        if len(detail_rows) == 0:
            return {"select_score": -1e18}
        detail_df = pd.DataFrame(detail_rows)
        return compute_selection_score_from_detail(detail_df)

    history_rows = []
    best_state = None
    best_score = -1e18
    best_epoch = -1
    bad_epochs = 0

    loop_start = max(LOOKBACK - 1, train_start_idx)

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_cnt = 0
        used_days = 0

        for t in range(loop_start, fit_end_idx + 1):
            mt = m[t]
            if mt.sum().item() < MIN_VALID_NODES_PER_DAY:
                continue

            seq = X[t - LOOKBACK + 1:t + 1].permute(1, 0, 2).contiguous()
            yt = y[t]
            node_valid = mt.float()

            nbr_i, nbr_w_msg, nbr_w_raw, deg_raw = get_graph_for_t(t)
            pred = model(seq, nbr_idx=nbr_i, nbr_w=nbr_w_msg, node_valid_mask=node_valid)
            if not torch.isfinite(pred[mt]).all():
                continue

            pred_t = pred[mt]
            yt_t = yt[mt]
            mse_loss = F.mse_loss(pred_t, yt_t)
            loss = mse_loss

            if USE_RANK_LOSS and RANK_LOSS_LAM > 0:
                rank_loss = ranking_loss_top_vs_rest(pred_t, yt_t)
                loss = loss + RANK_LOSS_LAM * rank_loss
            else:
                rank_loss = pred_t.new_tensor(0.0)

            if (nbr_i is not None) and (LAPLACIAN_LAM > 0):
                if USE_NORMALIZED_LAPLACIAN:
                    lap_loss = laplacian_loss_knn_normalized(pred, nbr_i, nbr_w_raw, deg_raw)
                else:
                    lap_loss = laplacian_loss_knn_unnorm(pred, nbr_i, nbr_w_raw)
                loss = loss + LAPLACIAN_LAM * lap_loss
            else:
                lap_loss = pred_t.new_tensor(0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP and GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            total_loss += float(loss.item()) * int(mt.sum().item())
            total_cnt += int(mt.sum().item())
            used_days += 1

        avg_loss = total_loss / max(total_cnt, 1)

        if use_validation:
            val_stat = evaluate_for_selection(val_start_idx, train_end_idx)
            val_score = float(val_stat["select_score"])
        else:
            val_stat = {"ret_at_1": np.nan, "ret_at_3": np.nan, "ret_at_5": np.nan, "ret_at_10": np.nan, "select_score": np.nan}
            val_score = -avg_loss

        history_rows.append({
            "epoch": ep,
            "train_loss": avg_loss,
            "val_select_score": val_score,
            "val_ret_at_1": val_stat.get("ret_at_1", np.nan),
            "val_ret_at_3": val_stat.get("ret_at_3", np.nan),
            "val_ret_at_5": val_stat.get("ret_at_5", np.nan),
            "val_ret_at_10": val_stat.get("ret_at_10", np.nan),
        })

        print(
            f"epoch {ep}/{EPOCHS} | train_loss={avg_loss:.6f} | used_obs={total_cnt} | used_days={used_days} | "
            f"val_score={val_score:.6f} | val@1={val_stat.get('ret_at_1', np.nan):.6f} | "
            f"val@3={val_stat.get('ret_at_3', np.nan):.6f} | val@5={val_stat.get('ret_at_5', np.nan):.6f} | "
            f"val@10={val_stat.get('ret_at_10', np.nan):.6f}"
        )

        if val_score > best_score:
            best_score = val_score
            best_epoch = ep
            bad_epochs = 0
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": ep,
                "val_score": val_score,
            }
        else:
            bad_epochs += 1

        if use_validation and bad_epochs >= EARLY_STOP_PATIENCE:
            print(f"[INFO] early stop at epoch={ep}, best_epoch={best_epoch}, best_val_score={best_score:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    history_df = pd.DataFrame(history_rows)
    split_info = {
        "fit_start_idx": train_start_idx,
        "fit_end_idx": fit_end_idx,
        "val_start_idx": val_start_idx if use_validation else None,
        "val_end_idx": train_end_idx if use_validation else None,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
    }
    return model, history_df, split_info


def evaluate_model_on_range(
    model,
    X_np, y_np, mask_np, dates, ts_codes_order,
    graph_dates, graph_nbr_idx, graph_nbr_w,
    device: torch.device,
    eval_start_idx: int, eval_end_idx: int,
    loss_curve: List[float]
):
    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    m = torch.from_numpy(mask_np).to(device)

    def graph_pos_for_pred_date(d_pred: pd.Timestamp):
        d = np.datetime64(d_pred.normalize().to_datetime64())
        pos = np.searchsorted(graph_dates, d) - int(GRAPH_SHIFT_TRADINGDAY)
        if pos < 0:
            return None
        return int(pos)

    def get_graph_for_t(t_idx: int):
        pos = graph_pos_for_pred_date(pd.Timestamp(dates[t_idx]))
        if pos is None or pos >= len(graph_dates):
            return None, None, None, None

        nbr_i = torch.from_numpy(graph_nbr_idx[pos].astype(np.int64)).to(device)
        nbr_w_raw = torch.from_numpy(graph_nbr_w[pos].astype(np.float32)).to(device)
        nbr_w_raw = torch.nan_to_num(nbr_w_raw, nan=0.0, posinf=0.0, neginf=0.0)
        nbr_w_raw = torch.clamp(nbr_w_raw, min=0.0)

        if USE_SYM_NORM_IN_GRAPHMIX:
            nbr_w_msg, deg_raw = knn_sym_norm_weights(nbr_i, nbr_w_raw)
        else:
            deg_raw = nbr_w_raw.sum(dim=1) + 1e-6
            nbr_w_msg = nbr_w_raw
        return nbr_i, nbr_w_msg, nbr_w_raw, deg_raw

    preds, trues = [], []
    daily_dates = []
    daily_mse = []
    daily_sse = []
    daily_nobs = []
    detail_rows = []

    model.eval()
    with torch.no_grad():
        loop_start = max(LOOKBACK - 1, eval_start_idx)
        for t in range(loop_start, eval_end_idx + 1):
            mt = m[t]
            if mt.sum().item() < MIN_VALID_NODES_PER_DAY:
                continue

            seq = X[t - LOOKBACK + 1:t + 1].permute(1, 0, 2).contiguous()
            yt = y[t]
            node_valid = mt.float()

            nbr_i, nbr_w_msg, _, _ = get_graph_for_t(t)
            pred = model(seq, nbr_idx=nbr_i, nbr_w=nbr_w_msg, node_valid_mask=node_valid)
            if not torch.isfinite(pred[mt]).all():
                continue

            err = pred[mt] - yt[mt]
            sse_day = float((err * err).sum().item())
            n_day = int(mt.sum().item())
            mse_day = sse_day / max(n_day, 1)

            current_date = pd.Timestamp(dates[t]).normalize()
            daily_dates.append(current_date)
            daily_sse.append(sse_day)
            daily_mse.append(mse_day)
            daily_nobs.append(n_day)

            pred_np = pred[mt].detach().cpu().numpy()
            true_np = yt[mt].detach().cpu().numpy()
            node_indices = np.where(mt.cpu().numpy())[0]

            preds.append(pred_np)
            trues.append(true_np)

            for node_idx, pred_val, true_val in zip(node_indices, pred_np, true_np):
                detail_rows.append({
                    "date": current_date,
                    "stock": ts_codes_order[node_idx],
                    "y_pred": float(pred_val),
                    "y_true": float(true_val),
                })

    if len(preds) == 0:
        return None

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    finite = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred = y_pred[finite]
    y_true = y_true[finite]

    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "mse": mse,
        "r2": r2,
        "n_obs": int(len(y_true)),
        "loss_curve": loss_curve,
        "daily_df": pd.DataFrame({
            "date": daily_dates,
            "mse": daily_mse,
            "sse": daily_sse,
            "nobs": daily_nobs,
        }),
        "detail_df": pd.DataFrame(detail_rows),
    }


def build_period_stats(detail_df: pd.DataFrame, period_freq: str, prefix: str):
    rows = []
    df = detail_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["period"] = df["date"].dt.to_period(period_freq).astype(str)

    for p, g in df.groupby("period"):
        yt = g["y_true"].to_numpy()
        yp = g["y_pred"].to_numpy()
        mse_ = float(mean_squared_error(yt, yp))
        r2_ = float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan
        rows.append({
            "period": p,
            f"mse_{prefix}": mse_,
            f"r2_{prefix}": r2_,
            "n_obs": int(len(g)),
        })
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


def build_period_compare(zero_detail: pd.DataFrame, real_detail: pd.DataFrame, period_freq: str):
    dz = zero_detail.copy()
    dr = real_detail.copy()

    dz["date"] = pd.to_datetime(dz["date"]).dt.normalize()
    dr["date"] = pd.to_datetime(dr["date"]).dt.normalize()
    dz["period"] = dz["date"].dt.to_period(period_freq).astype(str)
    dr["period"] = dr["date"].dt.to_period(period_freq).astype(str)

    rows_zero = []
    for p, g in dz.groupby("period"):
        yt = g["y_true"].to_numpy()
        yp = g["y_pred"].to_numpy()
        rows_zero.append({
            "period": p,
            "mse_zero": float(mean_squared_error(yt, yp)),
            "r2_zero": float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan,
            "n_obs_zero": int(len(g)),
        })
    stat_zero = pd.DataFrame(rows_zero)

    rows_real = []
    for p, g in dr.groupby("period"):
        yt = g["y_true"].to_numpy()
        yp = g["y_pred"].to_numpy()
        rows_real.append({
            "period": p,
            "mse_real": float(mean_squared_error(yt, yp)),
            "r2_real": float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan,
            "n_obs_real": int(len(g)),
        })
    stat_real = pd.DataFrame(rows_real)

    out = stat_zero.merge(stat_real, on="period", how="inner")
    out["delta_mse(real-zero)"] = out["mse_real"] - out["mse_zero"]
    out["delta_r2(real-zero)"] = out["r2_real"] - out["r2_zero"]
    out["n_obs"] = out[["n_obs_zero", "n_obs_real"]].min(axis=1)

    daily_zero = dz.groupby("date").agg(
        mse_zero=("y_true", lambda s: np.nan),
    )
    # safer / no apply deprecation
    d0_rows = []
    for d, g in dz.groupby("date", sort=True):
        yt = g["y_true"].to_numpy()
        yp = g["y_pred"].to_numpy()
        d0_rows.append({
            "date": pd.Timestamp(d),
            "mse_zero": float(mean_squared_error(yt, yp)),
            "sse_zero": float(np.sum((yt - yp) ** 2)),
            "nobs_zero": int(len(g)),
        })
    daily_zero = pd.DataFrame(d0_rows)

    d1_rows = []
    for d, g in dr.groupby("date", sort=True):
        yt = g["y_true"].to_numpy()
        yp = g["y_pred"].to_numpy()
        d1_rows.append({
            "date": pd.Timestamp(d),
            "mse_real": float(mean_squared_error(yt, yp)),
            "sse_real": float(np.sum((yt - yp) ** 2)),
            "nobs_real": int(len(g)),
        })
    daily_real = pd.DataFrame(d1_rows)

    daily_cmp = daily_zero.merge(daily_real, on="date", how="inner")
    daily_cmp["d_mse"] = daily_cmp["mse_zero"] - daily_cmp["mse_real"]
    daily_cmp["period"] = pd.to_datetime(daily_cmp["date"]).dt.to_period(period_freq).astype(str)

    dm_rows = []
    for p, g in daily_cmp.groupby("period"):
        dm_res = dm_test(g["d_mse"].to_numpy(), lag=DM_LAG, alternative=DM_ALTERNATIVE)
        sse0 = float(g["sse_zero"].sum())
        sser = float(g["sse_real"].sum())
        r2_os_vs_zero = float(1.0 - sser / max(sse0, 1e-12))
        dm_rows.append({
            "period": p,
            "dm_T": dm_res["T"],
            "dm_stat": dm_res["dm"],
            "dm_p_one": dm_res["p_one"],
            "dm_p_two": dm_res["p_two"],
            "dm_mean_d": dm_res["mean_d"],
            "dm_lag": dm_res["lag"],
            "r2_os_vs_zero": r2_os_vs_zero,
            "eval_days": int(len(g)),
        })

    dm_df = pd.DataFrame(dm_rows)
    out = out.merge(dm_df, on="period", how="left")
    return out.sort_values("period").reset_index(drop=True)


# =======================
# 7) 可视化
# =======================
def plot_trainlog(train_log_df: pd.DataFrame, out_png: str):
    if not _HAS_PLT or train_log_df is None or train_log_df.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(train_log_df["epoch"], train_log_df["train_loss"], marker="o", label="train_loss")
    plt.plot(train_log_df["epoch"], train_log_df["val_select_score"], marker="o", label="val_select_score")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Train log | {EXP_NAME}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_r2_compare(df_cmp: pd.DataFrame, zero_col: str, real_col: str, out_png: str, title: str):
    if not _HAS_PLT or df_cmp is None or df_cmp.empty:
        return
    x = np.arange(len(df_cmp))
    labels = df_cmp["period"].astype(str).tolist()
    width = 0.36
    plt.figure(figsize=(max(10, len(labels) * 0.65), 5))
    plt.bar(x - width / 2, df_cmp[zero_col], width=width, label=zero_col)
    plt.bar(x + width / 2, df_cmp[real_col], width=width, label=real_col)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("R²")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_delta_r2(df_cmp: pd.DataFrame, delta_col: str, out_png: str, title: str):
    if not _HAS_PLT or df_cmp is None or df_cmp.empty:
        return
    x = np.arange(len(df_cmp))
    labels = df_cmp["period"].astype(str).tolist()
    plt.figure(figsize=(max(10, len(labels) * 0.65), 5))
    plt.bar(x, df_cmp[delta_col])
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("ΔR² (Real - Zero)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# =======================
# 8) 主流程
# =======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)
    print("[INFO] PROFILE =", PROFILE)
    print("[INFO] EXP_NAME =", EXP_NAME)
    print("[INFO] Strong graph params retained | TOPK=", TOPK, "EDGE_TAU=", EDGE_TAU, "EDGE_POWER=", EDGE_POWER)
    print("[INFO] Ablation | LAPLACIAN_LAM=", LAPLACIAN_LAM, "USE_RANK_LOSS=", USE_RANK_LOSS, "RANK_LOSS_LAM=", RANK_LOSS_LAM)

    pack = np.load(CACHE_NPZ)
    X_np = pack["X_np"].astype(np.float32)
    y_np = pack["y_np"].astype(np.float32)
    mask_np = pack["mask_np"].astype(bool)

    with open(META_PKL, "rb") as f:
        meta = pickle.load(f)

    all_dates = [pd.Timestamp(d).normalize() for d in meta["all_dates"]]
    ts_codes_order = list(meta["ts_codes_order"])
    train_start_idx = int(meta["train_start_idx"])
    train_end_idx = int(meta["train_end_idx"])
    test_start_idx = int(meta["test_start_idx"])
    test_end_idx = int(meta["test_end_idx"])

    ret_mmap, ret_valid = build_or_load_returns_memmap(all_dates, ts_codes_order)
    g_dates, g_nbr_idx, g_nbr_w, g_node_valid = build_rolling_corr_graph(
        all_dates, ts_codes_order, ret_mmap, ret_valid
    )

    in_dim = X_np.shape[2]
    print(f"[INFO] tensor: T={X_np.shape[0]}, N={X_np.shape[1]}, F={X_np.shape[2]}")
    print(f"[INFO] graph cache ready: {GRAPH_CACHE}")

    print(f"\n===== Train {EXP_NAME} RealGraph =====")
    model, train_log_df, split_info = train_model_on_range(
        X_np, y_np, mask_np, all_dates, ts_codes_order,
        g_dates, g_nbr_idx, g_nbr_w,
        in_dim=in_dim,
        device=device,
        train_start_idx=train_start_idx,
        train_end_idx=train_end_idx,
        seed=SEED,
    )

    train_log_df.to_csv(os.path.join(RESULT_DIR, f"trainlog_{EXP_NAME}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(RESULT_DIR, f"splitinfo_{EXP_NAME}.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    train_eval = evaluate_model_on_range(
        model,
        X_np, y_np, mask_np, all_dates, ts_codes_order,
        g_dates, g_nbr_idx, g_nbr_w,
        device=device,
        eval_start_idx=train_start_idx,
        eval_end_idx=train_end_idx,
        loss_curve=train_log_df["train_loss"].tolist(),
    )

    test_eval = evaluate_model_on_range(
        model,
        X_np, y_np, mask_np, all_dates, ts_codes_order,
        g_dates, g_nbr_idx, g_nbr_w,
        device=device,
        eval_start_idx=test_start_idx,
        eval_end_idx=test_end_idx,
        loss_curve=train_log_df["train_loss"].tolist(),
    )

    if train_eval is None or test_eval is None:
        raise RuntimeError(f"{EXP_NAME} 评估为空，请检查数据覆盖。")

    train_detail = train_eval["detail_df"].copy()
    test_detail = test_eval["detail_df"].copy()

    train_quarter = build_period_stats(train_detail, "Q", "real")
    test_month = build_period_stats(test_detail, "M", "real")

    summary_df = pd.DataFrame([{
        "model": f"{EXP_NAME}_RealGraph",
        "profile": PROFILE,
        "graph_cache": GRAPH_CACHE,
        "mse_train": train_eval["mse"],
        "r2_train": train_eval["r2"],
        "mse_test": test_eval["mse"],
        "r2_test": test_eval["r2"],
        "n_obs_train": train_eval["n_obs"],
        "n_obs_test": test_eval["n_obs"],
        "best_epoch": split_info["best_epoch"],
        "best_val_score": split_info["best_val_score"],
        "fit_end_idx": split_info["fit_end_idx"],
        "val_start_idx": split_info["val_start_idx"],
        "val_end_idx": split_info["val_end_idx"],
        "topk_graph": TOPK,
        "edge_tau": EDGE_TAU,
        "edge_power": EDGE_POWER,
        "laplacian_lam": LAPLACIAN_LAM,
        "use_rank_loss": int(USE_RANK_LOSS),
        "rank_loss_lam": RANK_LOSS_LAM,
    }])

    train_detail.to_csv(os.path.join(RESULT_DIR, f"predictions_{EXP_NAME}_train_detail.csv"), index=False, encoding="utf-8-sig")
    test_detail.to_csv(os.path.join(RESULT_DIR, f"predictions_{EXP_NAME}_test_detail.csv"), index=False, encoding="utf-8-sig")
    train_eval["daily_df"].to_csv(os.path.join(RESULT_DIR, f"daily_{EXP_NAME}_train.csv"), index=False, encoding="utf-8-sig")
    test_eval["daily_df"].to_csv(os.path.join(RESULT_DIR, f"daily_{EXP_NAME}_test.csv"), index=False, encoding="utf-8-sig")
    train_quarter.to_csv(os.path.join(RESULT_DIR, f"train_{EXP_NAME}_by_quarter.csv"), index=False, encoding="utf-8-sig")
    test_month.to_csv(os.path.join(RESULT_DIR, f"test_{EXP_NAME}_by_month.csv"), index=False, encoding="utf-8-sig")
    summary_df.to_csv(os.path.join(RESULT_DIR, f"summary_{EXP_NAME}.csv"), index=False, encoding="utf-8-sig")

    plot_trainlog(train_log_df, os.path.join(PLOT_DIR, f"trainlog_{EXP_NAME}.png"))

    if os.path.exists(ZERO_TRAIN_DETAIL) and os.path.exists(ZERO_TEST_DETAIL):
        zero_train = pd.read_csv(ZERO_TRAIN_DETAIL)
        zero_test = pd.read_csv(ZERO_TEST_DETAIL)

        train_compare = build_period_compare(zero_train, train_detail, "Q")
        test_compare = build_period_compare(zero_test, test_detail, "M")

        train_compare.to_csv(os.path.join(RESULT_DIR, f"compare_{EXP_NAME}_vs_Zero_train_by_quarter.csv"), index=False, encoding="utf-8-sig")
        test_compare.to_csv(os.path.join(RESULT_DIR, f"compare_{EXP_NAME}_vs_Zero_test_by_month.csv"), index=False, encoding="utf-8-sig")

        plot_r2_compare(
            train_compare, "r2_zero", "r2_real",
            os.path.join(PLOT_DIR, "r2_train_quarter_compare.png"),
            f"Train quarter R² compare | {EXP_NAME} vs Zero"
        )
        plot_r2_compare(
            test_compare, "r2_zero", "r2_real",
            os.path.join(PLOT_DIR, "r2_test_month_compare.png"),
            f"Test month R² compare | {EXP_NAME} vs Zero"
        )
        plot_delta_r2(
            train_compare, "delta_r2(real-zero)",
            os.path.join(PLOT_DIR, "delta_r2_train_quarter.png"),
            f"Train quarter ΔR² | {EXP_NAME} - Zero"
        )
        plot_delta_r2(
            test_compare, "delta_r2(real-zero)",
            os.path.join(PLOT_DIR, "delta_r2_test_month.png"),
            f"Test month ΔR² | {EXP_NAME} - Zero"
        )

    print(f"\n完成：{EXP_NAME} strong-graph ablation RealGraph 已保存到 {RESULT_DIR}")
    print(f"图表已保存到: {PLOT_DIR}")
    print(summary_df)


if __name__ == "__main__":
    main()
