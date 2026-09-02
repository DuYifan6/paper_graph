# -*- coding: utf-8 -*-
"""
FiveMin-NoGraph parameter-matched control under the final common-MSE protocol.

Purpose
-------
Add only within-stock lagged 5-minute summaries to the common temporal model,
without graph construction or cross-stock aggregation.

Matched to Graph-2:
- Same LSTM / prediction head class
- Same LOOKBACK, HIDDEN, DROPOUT, LR, WEIGHT_DECAY, EPOCHS, GRAD_CLIP
- Same fixed fitting/validation/test split and early-stopping patience
- MSE-only training and minimum pooled validation-MSE checkpoint
- Same random seed convention

No graph information:
- nbr_idx=None
- nbr_w=None
- GraphMix is bypassed by design
- No Laplacian term

Interpretation
--------------
The self-only high-frequency branch has exactly 82,304 registered and active
parameters, matching GraphMix; total trainable parameters are 170,113.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import glob
import hashlib
import math
import json
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from project_config import MINUTE5_DIR as CONFIG_MINUTE5_DIR, OUTPUT_ROOT, common_cache_paths


# =======================
# 0) 配置
# =======================
CACHE_NPZ, META_PKL = map(str, common_cache_paths())

EXP_NAME = "FiveMin_NoGraph_ParameterMatched"

SANITY_ONLY = False
SANITY_EPOCHS = 1

LOOKBACK = 20

# ===== 5-minute self-only information =====
MINUTE5_DIR = str(CONFIG_MINUTE5_DIR)
MINUTE5_SUFFIX_CANDIDATES = ["_min.csv", "_5m.csv", "_5分钟.csv", ".csv"]
TIME_COL = "trade_time"
CLOSE_COL = "close"

BARS_PER_DAY = 49
RET_PER_DAY = BARS_PER_DAY - 1
ROLL_W = 20
CLIP_RET = 0.08

# 48 mean + 48 std + 48 latest-day normalized returns
HF_DIM = 144
HF_MLP_HIDDEN = 301

HIDDEN = 128
NUM_LAYERS = 1
DROPOUT = 0.1
LR = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
GRAD_CLIP = 1.0
GATE_HIDDEN = 128

MIN_VALID_NODES_PER_DAY = 50
DM_LAG = 2
DM_ALTERNATIVE = "greater"

# ===== fixed validation / selection =====
VAL_START_DATE = "2024-06-06"
EARLY_STOP_PATIENCE = 4

OUT_DIR = str(OUTPUT_ROOT / "five_min_nograph")
RESULT_DIR = os.path.join(OUT_DIR, "results")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
CACHE_DIR = os.path.join(OUT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

RET_MEMMAP_PATH = None
RET_VALID_PATH = None
HF_MEMMAP_PATH = None
DATA_CACHE_KEY = None


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


def resolve_split(dates, meta):
    """Return the fixed common fit/validation/OOS split."""
    train_start = int(meta["train_start_idx"])
    train_end = int(meta["train_end_idx"])
    test_start = int(meta["test_start_idx"])
    test_end = int(meta["test_end_idx"])

    date_arr = pd.DatetimeIndex(dates)
    candidates = np.flatnonzero(
        (date_arr >= pd.Timestamp(VAL_START_DATE))
        & (np.arange(len(date_arr)) >= train_start)
        & (np.arange(len(date_arr)) <= train_end)
    )
    if len(candidates) == 0:
        raise ValueError(f"Cannot locate validation start {VAL_START_DATE}.")
    val_start = int(candidates[0])
    fit_end = val_start - 1
    if fit_end < max(train_start, LOOKBACK - 1):
        raise ValueError("Fitting interval is too short for the 20-day lookback.")
    return train_start, fit_end, val_start, train_end, test_start, test_end



# =======================
# 2) 5-minute self-only information
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

    if ret.size != RET_PER_DAY or not np.all(np.isfinite(ret)):
        return None
    if CLIP_RET and CLIP_RET > 0:
        ret = np.clip(ret, -float(CLIP_RET), float(CLIP_RET))
    return ret


def build_or_load_returns_memmap(dates: List[pd.Timestamp], codes: List[str]):
    """
    Reuse Graph-2's already-built 5-minute return cache when available.
    Otherwise build the same [T,N,48] log-return cache from raw 5-minute closes.
    """
    global RET_MEMMAP_PATH, RET_VALID_PATH, DATA_CACHE_KEY

    T = len(dates)
    N = len(codes)
    M = RET_PER_DAY

    identity = "\n".join(
        [*(str(pd.Timestamp(d).date()) for d in dates), "--stocks--", *codes]
    )
    DATA_CACHE_KEY = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]

    RET_MEMMAP_PATH = os.path.join(
        CACHE_DIR, f"ret5m_logret_T{T}_N{N}_M{M}_{DATA_CACHE_KEY}.dat"
    )
    RET_VALID_PATH = os.path.join(
        CACHE_DIR, f"ret5m_valid_T{T}_N{N}_{DATA_CACHE_KEY}.npy"
    )

    if os.path.exists(RET_MEMMAP_PATH) and os.path.exists(RET_VALID_PATH):
        ret = np.memmap(
            RET_MEMMAP_PATH, dtype="float32", mode="r", shape=(T, N, M)
        )
        valid = np.load(RET_VALID_PATH).astype(np.uint8)
        print("[INFO] reuse local 5m return cache:", RET_MEMMAP_PATH)
        return ret, valid

    if not os.path.isdir(MINUTE5_DIR):
        raise FileNotFoundError(f"MINUTE5_DIR not found: {MINUTE5_DIR}")

    print("[INFO] building 5m log-return cache from raw files...")
    date_to_t = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
    date_set = set(date_to_t.keys())
    code_to_n = {c: i for i, c in enumerate(codes)}

    ret = np.memmap(
        RET_MEMMAP_PATH, dtype="float32", mode="w+", shape=(T, N, M)
    )
    ret[:] = 0.0
    valid = np.zeros((T, N), dtype=np.uint8)

    missing_files = 0

    for code in codes:
        fpath = find_5m_file(code)
        if fpath is None:
            missing_files += 1
            continue

        ni = code_to_n[code]
        try:
            df = pd.read_csv(
                fpath, usecols=[TIME_COL, CLOSE_COL], low_memory=False
            )
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

        df[CLOSE_COL] = pd.to_numeric(
            df[CLOSE_COL], errors="coerce"
        ).astype(np.float32)
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

    print(
        f"[INFO] returns built | missing_files={missing_files} | "
        f"saved={RET_MEMMAP_PATH}"
    )
    return np.memmap(
        RET_MEMMAP_PATH, dtype="float32", mode="r", shape=(T, N, M)
    ), valid


def build_or_load_self_hf_features(
    ret_mmap: np.memmap,
    valid: np.ndarray,
    T: int,
    N: int,
):
    """
    Build a disk-backed [T,N,144] feature tensor.

    For target day t, ONLY ret[t-ROLL_W:t, stock, :] is used.
    Hence the information window matches Graph-2's graph-construction window.

    No statistic is ever computed across stocks.
    """
    global HF_MEMMAP_PATH
    if DATA_CACHE_KEY is None:
        raise RuntimeError("Return-cache identity is unavailable.")

    HF_MEMMAP_PATH = os.path.join(
        CACHE_DIR,
        f"self5m_features_T{T}_N{N}_D{HF_DIM}_W{ROLL_W}_{DATA_CACHE_KEY}.dat",
    )
    valid_path = os.path.join(
        CACHE_DIR,
        f"self5m_feature_valid_T{T}_N{N}_W{ROLL_W}_{DATA_CACHE_KEY}.npy",
    )

    if os.path.exists(HF_MEMMAP_PATH) and os.path.exists(valid_path):
        hf = np.memmap(
            HF_MEMMAP_PATH, dtype="float32", mode="r",
            shape=(T, N, HF_DIM)
        )
        hf_valid = np.load(valid_path).astype(np.uint8)
        print("[INFO] reuse self-only 5m feature cache:", HF_MEMMAP_PATH)
        return hf, hf_valid

    print("[INFO] building self-only 5m feature cache (no cross-stock operations)...")
    hf = np.memmap(
        HF_MEMMAP_PATH, dtype="float32", mode="w+",
        shape=(T, N, HF_DIM)
    )
    hf[:] = 0.0
    hf_valid = np.zeros((T, N), dtype=np.uint8)

    for t in range(T):
        if t < ROLL_W:
            continue

        # Exact same 20-day raw-return window used by Graph-2 at graph date t.
        win = np.asarray(
            ret_mmap[t - ROLL_W:t, :, :], dtype=np.float32
        )  # [20,N,48]
        win_valid = valid[t - ROLL_W:t, :].astype(bool)
        ok = win_valid.all(axis=0)
        hf_valid[t] = ok.astype(np.uint8)

        if not ok.any():
            continue

        # Reshape to [N,20,48]. Normalize EACH STOCK using only its own history.
        z = win.transpose(1, 0, 2).copy()
        flat = z.reshape(N, ROLL_W * RET_PER_DAY)

        mu = flat.mean(axis=1, keepdims=True)
        sd = flat.std(axis=1, keepdims=True) + 1e-6
        flat = (flat - mu) / sd
        flat = np.clip(flat, -8.0, 8.0)
        flat[~ok] = 0.0

        z = flat.reshape(N, ROLL_W, RET_PER_DAY)

        # 144-dimensional deterministic self-only summary.
        intraday_mean = z.mean(axis=1)       # [N,48]
        intraday_std = z.std(axis=1)         # [N,48]
        latest_day = z[:, -1, :]             # [N,48]

        feat = np.concatenate(
            [intraday_mean, intraday_std, latest_day],
            axis=1
        ).astype(np.float32)

        feat[~ok] = 0.0
        feat = np.nan_to_num(
            feat, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

        hf[t] = feat

        if (t + 1) % 100 == 0:
            print(
                f"[INFO] self5m feature cache {t+1}/{T} | "
                f"valid_nodes={int(ok.sum())}"
            )

    hf.flush()
    np.save(valid_path, hf_valid)

    print("[INFO] self-only 5m feature cache saved:", HF_MEMMAP_PATH)
    return np.memmap(
        HF_MEMMAP_PATH, dtype="float32", mode="r",
        shape=(T, N, HF_DIM)
    ), hf_valid



# =======================
# 2) 模型
# =======================
class FiveMinNoGraphLSTM(nn.Module):
    """
    Same temporal LSTM and prediction head as Graph-2.
    Replaces GraphMix with a self-only 5-minute information encoder.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)

        # 82,301 parameters.
        self.hf_encoder = nn.Sequential(
            nn.Linear(HF_DIM, HF_MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HF_MLP_HIDDEN, hidden),
            nn.ReLU(),
        )

        # Three scalar parameters -> HF branch total = 82,304,
        # exactly equal to Graph-2 GraphMix.
        self.temporal_scale = nn.Parameter(torch.ones(1))
        self.hf_scale = nn.Parameter(torch.ones(1))
        self.fusion_bias = nn.Parameter(torch.zeros(1))

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, X, hf_features):
        out, _ = self.lstm(X)
        z_temporal = self.dropout(out[:, -1, :])

        hf_z = self.hf_encoder(hf_features)

        # Purely within-stock fusion; no node ever reads another node's features.
        z = (
            self.temporal_scale * z_temporal
            + self.hf_scale * hf_z
            + self.fusion_bias
        )

        return self.head(z).squeeze(-1)


def parameter_count_report(model: nn.Module) -> pd.DataFrame:
    hf_encoder_n = sum(
        p.numel() for p in model.hf_encoder.parameters() if p.requires_grad
    )
    fusion_n = (
        model.temporal_scale.numel()
        + model.hf_scale.numel()
        + model.fusion_bias.numel()
    )

    rows = [
        {
            "module": "lstm",
            "registered_trainable_params": int(
                sum(p.numel() for p in model.lstm.parameters() if p.requires_grad)
            ),
        },
        {
            "module": "5m_self_encoder_plus_fusion",
            "registered_trainable_params": int(hf_encoder_n + fusion_n),
        },
        {
            "module": "head",
            "registered_trainable_params": int(
                sum(p.numel() for p in model.head.parameters() if p.requires_grad)
            ),
        },
    ]

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows.append({
        "module": "TOTAL",
        "registered_trainable_params": int(total)
    })

    return pd.DataFrame(rows)


def active_gradient_report(model: nn.Module) -> Dict[str, Any]:
    total = 0
    hf_total = 0
    hf_grad = 0
    per_param = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        n = int(p.numel())
        total += n
        grad_present = p.grad is not None

        is_hf = (
            name.startswith("hf_encoder.")
            or name in {"temporal_scale", "hf_scale", "fusion_bias"}
        )
        if is_hf:
            hf_total += n
            if grad_present:
                hf_grad += n

        per_param.append({
            "name": name,
            "numel": n,
            "grad_present": bool(grad_present),
        })

    return {
        "registered_total": int(total),
        "hf_branch_registered": int(hf_total),
        "hf_branch_grad_present": int(hf_grad),
        "per_parameter": per_param,
    }


# =======================
# 3) 训练 / 评估
# =======================
def train_model_on_range(
    X_np, y_np, mask_np, dates, ts_codes_order,
    hf_memmap, hf_valid_np,
    in_dim: int, device: torch.device,
    train_start_idx: int, fit_end_idx: int,
    val_start_idx: int, val_end_idx: int,
    seed: int = 0
):
    set_seed(seed)

    model = FiveMinNoGraphLSTM(
        in_dim=in_dim,
        hidden=HIDDEN,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    param_df = parameter_count_report(model)
    param_df.to_csv(
        os.path.join(RESULT_DIR, f"parameter_counts_{EXP_NAME}.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    print("\n[PARAMETER COUNT]")
    print(param_df.to_string(index=False))

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    m = torch.from_numpy(mask_np).to(device)

    def get_hf_for_t(t_idx: int):
        hf_np = np.array(
            hf_memmap[t_idx],
            dtype=np.float32,
            copy=True
        )
        return torch.from_numpy(hf_np).to(device)

    def validation_mse(eval_start_idx: int, eval_end_idx: int):
        model.eval()
        total_sse = 0.0
        total_n = 0

        with torch.no_grad():
            loop_start = max(LOOKBACK - 1, eval_start_idx)
            for t in range(loop_start, eval_end_idx + 1):
                mt = m[t]
                if mt.sum().item() < MIN_VALID_NODES_PER_DAY:
                    continue

                seq = X[t - LOOKBACK + 1:t + 1].permute(1, 0, 2).contiguous()
                hf_t = get_hf_for_t(t)
                pred = model(seq, hf_t)
                if not torch.isfinite(pred[mt]).all():
                    continue
                err = pred[mt] - y[t][mt]
                total_sse += float(torch.sum(err * err).item())
                total_n += int(mt.sum().item())
        return float(total_sse / total_n) if total_n > 0 else float("inf")

    history_rows = []
    best_state = None
    best_val_mse = float("inf")
    best_epoch = -1
    bad_epochs = 0

    loop_start = max(LOOKBACK - 1, train_start_idx)
    gradient_audit_written = False
    epochs_to_run = SANITY_EPOCHS if SANITY_ONLY else EPOCHS

    for ep in range(1, epochs_to_run + 1):
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

            hf_t = get_hf_for_t(t)
            pred = model(seq, hf_t)
            if not torch.isfinite(pred[mt]).all():
                continue

            pred_t = pred[mt]
            yt_t = yt[mt]

            loss = F.mse_loss(pred_t, yt_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if not gradient_audit_written:
                grad_report = active_gradient_report(model)
                audit_df = pd.DataFrame(grad_report["per_parameter"])
                audit_df.to_csv(
                    os.path.join(RESULT_DIR, f"gradient_audit_{EXP_NAME}.csv"),
                    index=False,
                    encoding="utf-8-sig"
                )

                print("\n[GRADIENT AUDIT]")
                print("Registered total params =", grad_report["registered_total"])
                print("5m self branch registered params =", grad_report["hf_branch_registered"])
                print("5m self branch params with grad =", grad_report["hf_branch_grad_present"])
                print("Expected Graph-2 matched total = 170113")
                print("Expected GraphMix-matched branch = 82304")

                if grad_report["registered_total"] != 170113:
                    raise RuntimeError(
                        f"Parameter match failed: total={grad_report['registered_total']} != 170113"
                    )
                if grad_report["hf_branch_registered"] != 82304:
                    raise RuntimeError(
                        f"5m branch match failed: {grad_report['hf_branch_registered']} != 82304"
                    )
                if grad_report["hf_branch_grad_present"] != 82304:
                    raise RuntimeError(
                        "5m branch did not fully participate in backward."
                    )

                gradient_audit_written = True

            if GRAD_CLIP and GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            total_loss += float(loss.item()) * int(mt.sum().item())
            total_cnt += int(mt.sum().item())
            used_days += 1

        avg_loss = total_loss / max(total_cnt, 1)

        val_mse = validation_mse(val_start_idx, val_end_idx)

        history_rows.append({
            "epoch": ep,
            "train_mse": avg_loss,
            "val_mse": val_mse,
            "used_obs": total_cnt,
            "used_days": used_days,
        })

        print(
            f"epoch {ep}/{epochs_to_run} | train_mse={avg_loss:.8f} | "
            f"val_mse={val_mse:.8f} | used_obs={total_cnt} | used_days={used_days}"
        )

        if val_mse < best_val_mse - 1e-8:
            best_val_mse = val_mse
            best_epoch = ep
            bad_epochs = 0
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": ep,
                "val_mse": val_mse,
            }
        else:
            bad_epochs += 1

        if bad_epochs >= EARLY_STOP_PATIENCE:
            print(
                f"[INFO] early stop at epoch={ep}, best_epoch={best_epoch}, "
                f"best_val_mse={best_val_mse:.8f}"
            )
            break

    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was selected.")
    model.load_state_dict(best_state["model"])

    history_df = pd.DataFrame(history_rows)
    split_info = {
        "fit_start_idx": train_start_idx,
        "fit_end_idx": fit_end_idx,
        "val_start_idx": val_start_idx,
        "val_end_idx": val_end_idx,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
    }

    return model, history_df, split_info


def evaluate_model_on_range(
    model,
    X_np, y_np, mask_np, dates, ts_codes_order,
    hf_memmap, hf_valid_np,
    device: torch.device,
    eval_start_idx: int, eval_end_idx: int,
    loss_curve: List[float]
):
    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    m = torch.from_numpy(mask_np).to(device)

    def get_hf_for_t(t_idx: int):
        hf_np = np.array(
            hf_memmap[t_idx],
            dtype=np.float32,
            copy=True
        )
        return torch.from_numpy(hf_np).to(device)

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

            hf_t = get_hf_for_t(t)
            pred = model(seq, hf_t)
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


# =======================
# 4) 主流程
# =======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)
    print("[INFO] EXP_NAME =", EXP_NAME)
    print("[INFO] SANITY_ONLY =", SANITY_ONLY)

    pack = np.load(CACHE_NPZ)
    X_np = pack["X_np"].astype(np.float32)
    y_np = pack["y_np"].astype(np.float32)
    mask_np = pack["mask_np"].astype(bool)

    with open(META_PKL, "rb") as f:
        meta = pickle.load(f)

    all_dates = [pd.Timestamp(d).normalize() for d in meta["all_dates"]]
    ts_codes_order = list(meta["ts_codes_order"])
    (
        train_start_idx,
        fit_end_idx,
        val_start_idx,
        train_end_idx,
        test_start_idx,
        test_end_idx,
    ) = resolve_split(all_dates, meta)
    print(
        f"[SPLIT] fit: {all_dates[max(train_start_idx, LOOKBACK - 1)].date()} "
        f".. {all_dates[fit_end_idx].date()}"
    )
    print(
        f"[SPLIT] validation: {all_dates[val_start_idx].date()} "
        f".. {all_dates[train_end_idx].date()}"
    )
    print(
        f"[SPLIT] test: {all_dates[test_start_idx].date()} "
        f".. {all_dates[test_end_idx].date()}"
    )

    in_dim = X_np.shape[2]
    T, N = X_np.shape[0], X_np.shape[1]
    print(f"[INFO] tensor: T={T}, N={N}, F={X_np.shape[2]}")

    ret_mmap, ret_valid = build_or_load_returns_memmap(
        all_dates, ts_codes_order
    )
    hf_memmap, hf_valid = build_or_load_self_hf_features(
        ret_mmap, ret_valid, T=T, N=N
    )

    print("\n===== Train 5m-NoGraph Parameter-Matched Control =====")
    print("[INFO] No graph / no correlation / no cross-stock aggregation")
    print("[INFO] 5m window for target t = returns[t-20:t]")
    print("[INFO] HF_DIM =", HF_DIM, "| HF_MLP_HIDDEN =", HF_MLP_HIDDEN)
    model, train_log_df, split_info = train_model_on_range(
        X_np, y_np, mask_np, all_dates, ts_codes_order,
        hf_memmap, hf_valid,
        in_dim=in_dim,
        device=device,
        train_start_idx=train_start_idx,
        fit_end_idx=fit_end_idx,
        val_start_idx=val_start_idx,
        val_end_idx=train_end_idx,
        seed=0
    )

    train_log_df.to_csv(
        os.path.join(RESULT_DIR, f"trainlog_{EXP_NAME}.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    with open(os.path.join(RESULT_DIR, f"splitinfo_{EXP_NAME}.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    val_eval = evaluate_model_on_range(
        model,
        X_np, y_np, mask_np, all_dates, ts_codes_order,
        hf_memmap, hf_valid,
        device=device,
        eval_start_idx=val_start_idx,
        eval_end_idx=train_end_idx,
        loss_curve=train_log_df["val_mse"].tolist()
    )

    test_eval = evaluate_model_on_range(
        model,
        X_np, y_np, mask_np, all_dates, ts_codes_order,
        hf_memmap, hf_valid,
        device=device,
        eval_start_idx=test_start_idx,
        eval_end_idx=test_end_idx,
        loss_curve=train_log_df["val_mse"].tolist()
    )

    if val_eval is None or test_eval is None:
        raise RuntimeError("5m-NoGraph evaluation is empty; check data coverage.")

    val_detail = val_eval["detail_df"].copy()
    test_detail = test_eval["detail_df"].copy()

    val_quarter = build_period_stats(val_detail, "Q", "five_min_nograph")
    test_month = build_period_stats(test_detail, "M", "five_min_nograph")

    summary_df = pd.DataFrame([{
        "model": EXP_NAME,
        "control_type": "self_5m_no_graph_parameter_matched",
        "hf_dim": HF_DIM,
        "hf_mlp_hidden": HF_MLP_HIDDEN,
        "hf_window_days": ROLL_W,
        "hf_ret_per_day": RET_PER_DAY,
        "cross_stock_operation": 0,
        "training_loss": "MSE",
        "ranking_loss": 0,
        "selection_criterion": "minimum pooled validation MSE",
        "registered_total_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "mse_validation": val_eval["mse"],
        "r2_validation": val_eval["r2"],
        "mse_test": test_eval["mse"],
        "r2_test": test_eval["r2"],
        "n_obs_validation": val_eval["n_obs"],
        "n_obs_test": test_eval["n_obs"],
        "best_epoch": split_info["best_epoch"],
        "best_val_mse": split_info["best_val_mse"],
        "fit_end_idx": split_info["fit_end_idx"],
        "val_start_idx": split_info["val_start_idx"],
        "val_end_idx": split_info["val_end_idx"],
    }])

    val_detail.to_csv(os.path.join(RESULT_DIR, "predictions_FiveMin_NoGraph_validation_detail.csv"), index=False, encoding="utf-8-sig")
    test_detail.to_csv(os.path.join(RESULT_DIR, "predictions_FiveMin_NoGraph_test_detail.csv"), index=False, encoding="utf-8-sig")
    val_eval["daily_df"].to_csv(os.path.join(RESULT_DIR, "daily_FiveMin_NoGraph_validation.csv"), index=False, encoding="utf-8-sig")
    test_eval["daily_df"].to_csv(os.path.join(RESULT_DIR, "daily_FiveMin_NoGraph_test.csv"), index=False, encoding="utf-8-sig")
    val_quarter.to_csv(os.path.join(RESULT_DIR, "validation_FiveMin_NoGraph_by_quarter.csv"), index=False, encoding="utf-8-sig")
    test_month.to_csv(os.path.join(RESULT_DIR, "test_FiveMin_NoGraph_by_month.csv"), index=False, encoding="utf-8-sig")
    summary_df.to_csv(os.path.join(RESULT_DIR, "summary_FiveMin_NoGraph.csv"), index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), os.path.join(RESULT_DIR, "best_FiveMin_NoGraph.pt"))

    print("\n完成：5m-NoGraph Parameter-Matched 已保存到", RESULT_DIR)
    print(summary_df)


if __name__ == "__main__":
    main()
