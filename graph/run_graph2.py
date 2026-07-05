# -*- coding: utf-8 -*-
"""
Graph2 内嵌版 RealGraph
- 读取 common_preprocess_fixedsplit.py 生成的公共缓存
- 不再依赖外部预生成 graph2_cache.npz
- 直接在本脚本里从 5min 数据构建【图二：滚动20天相关系数日图 -> 01化二值图】
- 然后训练一个 RealGraph
- 训练集按季度输出
- 测试集按月份输出
- 如果给出 ZeroGraph 预测文件，会额外生成 compare 表与 DM 检验

说明：
- 除了“图构造”不同外，其余框架与 Graph1 修改版保持一致
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


# =======================
# 0) 配置
# =======================
CACHE_NPZ = r"D:\PythonProject\LASSO_FINAL\common_preprocess_cache\common_tensor_cache.npz"
META_PKL  = r"D:\PythonProject\LASSO_FINAL\common_preprocess_cache\common_tensor_meta.pkl"

EXP_NAME = "Graph2"

# 如果存在固定 ZeroGraph 结果，则自动生成 compare 表
ZERO_TRAIN_DETAIL = r"D:\PythonProject\LASSO_FINAL\graph\fixed_zerograph\results\predictions_ZeroGraph_train_detail.csv"
ZERO_TEST_DETAIL  = r"D:\PythonProject\LASSO_FINAL\graph\fixed_zerograph\results\predictions_ZeroGraph_test_detail.csv"
# ===== 5min 数据目录 =====
MINUTE5_DIR = r"D:\Lasso\min5_aligned"
MINUTE5_SUFFIX_CANDIDATES = ["_min.csv", "_5m.csv", "_5分钟.csv", ".csv"]
TIME_COL = "trade_time"
CLOSE_COL = "close"

# bars/returns per day
BARS_PER_DAY = 49
RET_PER_DAY  = BARS_PER_DAY - 1

# ===== Graph-2 参数 =====
ROLL_W = 20
TOPK = 40

# 这里修正成 0：graph[t] 本身就是由 [t-20, ..., t-1] 构成，用来预测 y[t]
GRAPH_SHIFT_TRADINGDAY = 0

MARKET_REMOVE = True
CLIP_RET = 0.08
ABS_CORR = True

# 01化阈值
BIN_TAU = 0.00

REQUIRE_FULL_WINDOW = True
MIN_DAYS_IN_WINDOW = 16
GPU_BLOCK = 256   # 保留接口，Graph2 这里默认走 CPU exact

# ===== 模型参数（与 Graph1 修改版保持一致） =====
LOOKBACK = 20
HIDDEN = 128
NUM_LAYERS = 1
DROPOUT = 0.1
LR = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 15
GRAD_CLIP = 1.0
GATE_HIDDEN = 128

MIN_VALID_NODES_PER_DAY = 50

# 非图构造部分保持和 Graph1 修改版一致
LAPLACIAN_LAM = 1e-3
USE_SYM_NORM_IN_GRAPHMIX = False
USE_NORMALIZED_LAPLACIAN = False

DM_LAG = 2
DM_ALTERNATIVE = "greater"

# ===== 输出目录 =====
OUT_DIR = rf"D:\PythonProject\LASSO_FINAL\graph\graph2"
CACHE_DIR  = os.path.join(OUT_DIR, "cache")
GRAPH_DIR  = os.path.join(OUT_DIR, "graphs")
RESULT_DIR = os.path.join(OUT_DIR, "results")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

RET_MEMMAP_PATH = None
RET_VALID_PATH  = None

GRAPH_CACHE = os.path.join(
    GRAPH_DIR,
    f"roll{ROLL_W}_corr_BINARY01_abs{int(ABS_CORR)}_mkt{int(MARKET_REMOVE)}_topk{TOPK}"
    f"_binTau{BIN_TAU}_full{int(REQUIRE_FULL_WINDOW)}.npz"
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


# =======================
# 2) 5min 文件读取与收益构建
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
# 3) Graph-2：CPU 精确相关 TopK
# =======================
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
    """
    directed KNN -> undirected union -> per-row topk
    无自环；不足位置用 (self, 0) padding
    """
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


# =======================
# 4) Graph-2：构建二值图
# =======================
def build_graph2_binary01(dates: List[pd.Timestamp], codes: List[str], ret_mmap: np.memmap, valid: np.ndarray):
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
    nbr_w_all   = np.zeros((T, N, topk_eff), dtype=np.float32)
    node_valid  = np.zeros((T, N), dtype=np.uint8)

    print("[INFO] 构图二：rolling corr -> TopK -> 01化 -> 无向 -> 无自环")
    print(f"[INFO] T={T} N={N} window={ROLL_W} L={L} TOPK={topk_eff} market_remove={MARKET_REMOVE} require_full={REQUIRE_FULL_WINDOW} BIN_TAU={BIN_TAU}")

    dates_np = pd.to_datetime(pd.Series(dates)).to_numpy(dtype="datetime64[ns]")

    for t in tqdm(range(T), desc="build Graph-2 (binary01)"):
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
        sd = X.std(axis=1, ddof=1, keepdims=True) + 1e-6
        X = (X - mu) / sd

        X[~ok] = 0.0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        idx, val = corr_topk_cpu_exact(X, topk_eff)

        val = np.clip(val, 0.0, 1.0)
        if BIN_TAU and BIN_TAU > 0:
            val = (val >= float(BIN_TAU)).astype(np.float32)
        else:
            val = (val > 0).astype(np.float32)

        bad = ~ok
        if bad.any():
            idx[bad] = np.arange(N, dtype=np.int64)[bad][:, None]
            val[bad] = 0.0

        idx_sym, val_sym = symmetrize_union_and_retopk(idx, val, topk_eff)
        val_sym = (val_sym > 0).astype(np.float32)

        nbr_idx_all[t] = idx_sym.astype(np.uint16)
        nbr_w_all[t]   = val_sym.astype(np.float32)

    meta = dict(
        graph_type="Graph-2_binary01_rolling_abs_corr_from_5m_logret",
        roll_window_days=ROLL_W,
        ret_per_day=RET_PER_DAY,
        topk=topk_eff,
        abs_corr=bool(ABS_CORR),
        market_remove=bool(MARKET_REMOVE),
        clip_ret=float(CLIP_RET),
        binary01=True,
        bin_tau=float(BIN_TAU),
        undirected_union=True,
        no_self_loop=True,
        padding_self_weight0=True,
        note="Graph(day t) uses window [t-ROLL_W, t-1]. Predict y(t) uses graph(t).",
        weight_values="{0,1}"
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
# 5) 图工具
# =======================
def laplacian_loss_knn_binary(pred: torch.Tensor, nbr_idx: torch.Tensor, nbr_w: torch.Tensor):
    nbr_w = torch.nan_to_num(nbr_w, nan=0.0, posinf=0.0, neginf=0.0)
    nbr_w = torch.clamp(nbr_w, 0.0, 1.0)
    nbr_pred = pred.index_select(0, nbr_idx.reshape(-1)).view_as(nbr_w)
    diff = pred.view(-1, 1) - nbr_pred
    return (nbr_w * diff * diff).mean()


# =======================
# 6) 模型
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
        nbr_w = torch.clamp(nbr_w, 0.0, 1.0)

        Z_nb_raw = Z_send.index_select(0, nbr_idx.reshape(-1)).view(N, K, H)
        w = nbr_w.view(N, K, 1)

        num = (w * Z_nb_raw).sum(dim=1)
        den = nbr_w.sum(dim=1, keepdim=True)
        Z_nb = num / (den + 1e-6)

        no_edge = (den < 1e-6).float()
        cat = torch.cat([Z, Z_nb], dim=1)
        g = torch.sigmoid(self.lin_gate2(F.relu(self.lin_gate(cat))))
        g = g * (1.0 - no_edge)
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
# 7) 训练 / 评估
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

    def graph_pos_for_pred_date(d_pred: pd.Timestamp):
        d = np.datetime64(d_pred.normalize().to_datetime64())
        pos = np.searchsorted(graph_dates, d) - int(GRAPH_SHIFT_TRADINGDAY)
        if pos < 0:
            return None
        return int(pos)

    def get_graph_for_t(t_idx: int):
        pos = graph_pos_for_pred_date(pd.Timestamp(dates[t_idx]))
        if pos is None or pos >= len(graph_dates):
            return None, None

        nbr_i = torch.from_numpy(graph_nbr_idx[pos].astype(np.int64)).to(device)

        nbr_w = torch.from_numpy(graph_nbr_w[pos].astype(np.float32)).to(device)
        nbr_w = torch.nan_to_num(nbr_w, nan=0.0, posinf=0.0, neginf=0.0)
        nbr_w = torch.clamp(nbr_w, 0.0, 1.0)

        return nbr_i, nbr_w

    loss_curve = []
    loop_start = max(LOOKBACK - 1, train_start_idx)

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_cnt = 0
        used_days = 0

        for t in range(loop_start, train_end_idx + 1):
            mt = m[t]
            if mt.sum().item() < MIN_VALID_NODES_PER_DAY:
                continue

            seq = X[t - LOOKBACK + 1:t + 1].permute(1, 0, 2).contiguous()
            yt = y[t]
            node_valid = mt.float()

            nbr_i, nbr_w = get_graph_for_t(t)

            pred = model(seq, nbr_idx=nbr_i, nbr_w=nbr_w, node_valid_mask=node_valid)
            if not torch.isfinite(pred[mt]).all():
                continue

            loss = F.mse_loss(pred[mt], yt[mt])

            if (nbr_i is not None) and (LAPLACIAN_LAM > 0):
                loss = loss + LAPLACIAN_LAM * laplacian_loss_knn_binary(pred, nbr_i, nbr_w)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP and GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            total_loss += float(loss.item()) * int(mt.sum().item())
            total_cnt += int(mt.sum().item())
            used_days += 1

        avg_loss = total_loss / max(total_cnt, 1)
        loss_curve.append(avg_loss)
        print(f"epoch {ep}/{EPOCHS} | avg_loss={avg_loss:.6f} | used_obs={total_cnt} | used_days={used_days}")

    return model, loss_curve


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
            return None, None

        nbr_i = torch.from_numpy(graph_nbr_idx[pos].astype(np.int64)).to(device)

        nbr_w = torch.from_numpy(graph_nbr_w[pos].astype(np.float32)).to(device)
        nbr_w = torch.nan_to_num(nbr_w, nan=0.0, posinf=0.0, neginf=0.0)
        nbr_w = torch.clamp(nbr_w, 0.0, 1.0)

        return nbr_i, nbr_w

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

            nbr_i, nbr_w = get_graph_for_t(t)

            pred = model(seq, nbr_idx=nbr_i, nbr_w=nbr_w, node_valid_mask=node_valid)
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

    daily_zero = dz.groupby("date").apply(
        lambda g: pd.Series({
            "mse_zero": mean_squared_error(g["y_true"], g["y_pred"]),
            "sse_zero": np.sum((g["y_true"] - g["y_pred"]) ** 2),
            "nobs_zero": len(g)
        })
    ).reset_index()

    daily_real = dr.groupby("date").apply(
        lambda g: pd.Series({
            "mse_real": mean_squared_error(g["y_true"], g["y_pred"]),
            "sse_real": np.sum((g["y_true"] - g["y_pred"]) ** 2),
            "nobs_real": len(g)
        })
    ).reset_index()

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
# 8) 主流程
# =======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)
    print("[INFO] EXP_NAME =", EXP_NAME)

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

    # ===== 这里替代原来的 load_graph_cache(GRAPH_CACHE) =====
    ret_mmap, ret_valid = build_or_load_returns_memmap(all_dates, ts_codes_order)
    g_dates, g_nbr_idx, g_nbr_w, g_node_valid = build_graph2_binary01(
        all_dates, ts_codes_order, ret_mmap, ret_valid
    )

    in_dim = X_np.shape[2]
    print(f"[INFO] tensor: T={X_np.shape[0]}, N={X_np.shape[1]}, F={X_np.shape[2]}")
    print(f"[INFO] graph cache ready: {GRAPH_CACHE}")

    print(f"\n===== Train {EXP_NAME} RealGraph =====")
    model, loss_curve = train_model_on_range(
        X_np, y_np, mask_np, all_dates, ts_codes_order,
        g_dates, g_nbr_idx, g_nbr_w,
        in_dim=in_dim,
        device=device,
        train_start_idx=train_start_idx,
        train_end_idx=train_end_idx,
        seed=0
    )

    train_eval = evaluate_model_on_range(
        model,
        X_np, y_np, mask_np, all_dates, ts_codes_order,
        g_dates, g_nbr_idx, g_nbr_w,
        device=device,
        eval_start_idx=train_start_idx,
        eval_end_idx=train_end_idx,
        loss_curve=loss_curve
    )

    test_eval = evaluate_model_on_range(
        model,
        X_np, y_np, mask_np, all_dates, ts_codes_order,
        g_dates, g_nbr_idx, g_nbr_w,
        device=device,
        eval_start_idx=test_start_idx,
        eval_end_idx=test_end_idx,
        loss_curve=loss_curve
    )

    if train_eval is None or test_eval is None:
        raise RuntimeError(f"{EXP_NAME} 评估为空，请检查数据覆盖。")

    train_detail = train_eval["detail_df"].copy()
    test_detail = test_eval["detail_df"].copy()

    train_quarter = build_period_stats(train_detail, "Q", "real")
    test_month = build_period_stats(test_detail, "M", "real")

    summary_df = pd.DataFrame([{
        "model": f"{EXP_NAME}_RealGraph",
        "graph_cache": GRAPH_CACHE,
        "mse_train": train_eval["mse"],
        "r2_train": train_eval["r2"],
        "mse_test": test_eval["mse"],
        "r2_test": test_eval["r2"],
        "n_obs_train": train_eval["n_obs"],
        "n_obs_test": test_eval["n_obs"],
    }])

    train_detail.to_csv(os.path.join(RESULT_DIR, f"predictions_{EXP_NAME}_train_detail.csv"), index=False, encoding="utf-8-sig")
    test_detail.to_csv(os.path.join(RESULT_DIR, f"predictions_{EXP_NAME}_test_detail.csv"), index=False, encoding="utf-8-sig")
    train_eval["daily_df"].to_csv(os.path.join(RESULT_DIR, f"daily_{EXP_NAME}_train.csv"), index=False, encoding="utf-8-sig")
    test_eval["daily_df"].to_csv(os.path.join(RESULT_DIR, f"daily_{EXP_NAME}_test.csv"), index=False, encoding="utf-8-sig")
    train_quarter.to_csv(os.path.join(RESULT_DIR, f"train_{EXP_NAME}_by_quarter.csv"), index=False, encoding="utf-8-sig")
    test_month.to_csv(os.path.join(RESULT_DIR, f"test_{EXP_NAME}_by_month.csv"), index=False, encoding="utf-8-sig")
    summary_df.to_csv(os.path.join(RESULT_DIR, f"summary_{EXP_NAME}.csv"), index=False, encoding="utf-8-sig")

    # 如果固定 ZeroGraph 结果存在，就额外生成 compare 表
    if os.path.exists(ZERO_TRAIN_DETAIL) and os.path.exists(ZERO_TEST_DETAIL):
        zero_train = pd.read_csv(ZERO_TRAIN_DETAIL)
        zero_test = pd.read_csv(ZERO_TEST_DETAIL)

        train_compare = build_period_compare(zero_train, train_detail, "Q")
        test_compare = build_period_compare(zero_test, test_detail, "M")

        train_compare.to_csv(os.path.join(RESULT_DIR, f"compare_{EXP_NAME}_vs_Zero_train_by_quarter.csv"), index=False, encoding="utf-8-sig")
        test_compare.to_csv(os.path.join(RESULT_DIR, f"compare_{EXP_NAME}_vs_Zero_test_by_month.csv"), index=False, encoding="utf-8-sig")

    print(f"\n完成：{EXP_NAME} RealGraph 已保存到 {RESULT_DIR}")
    print(summary_df)


if __name__ == "__main__":
    main()