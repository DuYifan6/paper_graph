# -*- coding: utf-8 -*-
"""
ZeroGraph baseline（validation 选模版）
- 读取 common_preprocess_fixedsplit.py 生成的公共缓存
- 不使用任何图输入（nbr_idx=None, nbr_w=None）
- 训练集按季度输出
- 测试集按月份输出
- 增加 validation 切分 / best checkpoint / early stopping
- validation 选模标准与 Real 版保持一致：按 K=1/3/5/10 的联合指标选最优 epoch

建议用途：
- 作为 Real-A / Real-B / Real-C 的公平对照基线
- 不建议在这版里再加 graph 或 Laplacian
"""

import os
import math
import json
import pickle
import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F


# =======================
# 0) 配置
# =======================
CACHE_NPZ = r"D:\PythonProject\LASSO_FINAL\common_preprocess_cache\common_tensor_cache.npz"
META_PKL  = r"D:\PythonProject\LASSO_FINAL\common_preprocess_cache\common_tensor_meta.pkl"

EXP_NAME = "ZeroGraph_valselect"

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
DM_LAG = 2
DM_ALTERNATIVE = "greater"

# ===== validation / 选模 =====
VALID_RATIO = 0.20
EARLY_STOP_PATIENCE = 4

SELECT_KS = [1, 3, 5, 10]
SELECT_WEIGHTS = {1: 0.10, 3: 0.40, 5: 0.30, 10: 0.20}

# ===== ranking loss（默认关闭，保持纯 Zero baseline） =====
USE_RANK_LOSS = False
RANK_LOSS_LAM = 0.0
RANK_TOP_FRAC = 0.20
RANK_MIN_TOP = 5
RANK_MAX_TOP = 50

OUT_DIR = r"D:\PythonProject\LASSO_FINAL\graph_调整\fixed_zerograph_valselect"
RESULT_DIR = os.path.join(OUT_DIR, "results")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


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
    loss = F.softplus(-(p_top - p_rest)).mean()
    return loss


# =======================
# 2) 模型
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
# 3) 训练 / 评估
# =======================
def train_model_on_range(
    X_np, y_np, mask_np, dates, ts_codes_order,
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

                pred = model(seq, nbr_idx=None, nbr_w=None, node_valid_mask=node_valid)
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

            pred = model(seq, nbr_idx=None, nbr_w=None, node_valid_mask=node_valid)
            if not torch.isfinite(pred[mt]).all():
                continue

            pred_t = pred[mt]
            yt_t = yt[mt]

            mse_loss = F.mse_loss(pred_t, yt_t)
            loss = mse_loss

            if USE_RANK_LOSS and RANK_LOSS_LAM > 0:
                rank_loss = ranking_loss_top_vs_rest(pred_t, yt_t)
                loss = loss + RANK_LOSS_LAM * rank_loss

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
    device: torch.device,
    eval_start_idx: int, eval_end_idx: int,
    loss_curve: List[float]
):
    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    m = torch.from_numpy(mask_np).to(device)

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

            pred = model(seq, nbr_idx=None, nbr_w=None, node_valid_mask=node_valid)
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

    in_dim = X_np.shape[2]
    print(f"[INFO] tensor: T={X_np.shape[0]}, N={X_np.shape[1]}, F={X_np.shape[2]}")

    print("\n===== Train ZeroGraph baseline (validation-select) =====")
    model, train_log_df, split_info = train_model_on_range(
        X_np, y_np, mask_np, all_dates, ts_codes_order,
        in_dim=in_dim,
        device=device,
        train_start_idx=train_start_idx,
        train_end_idx=train_end_idx,
        seed=0
    )

    train_log_df.to_csv(
        os.path.join(RESULT_DIR, f"trainlog_{EXP_NAME}.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    with open(os.path.join(RESULT_DIR, f"splitinfo_{EXP_NAME}.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    train_eval = evaluate_model_on_range(
        model,
        X_np, y_np, mask_np, all_dates, ts_codes_order,
        device=device,
        eval_start_idx=train_start_idx,
        eval_end_idx=train_end_idx,
        loss_curve=train_log_df["train_loss"].tolist()
    )

    test_eval = evaluate_model_on_range(
        model,
        X_np, y_np, mask_np, all_dates, ts_codes_order,
        device=device,
        eval_start_idx=test_start_idx,
        eval_end_idx=test_end_idx,
        loss_curve=train_log_df["train_loss"].tolist()
    )

    if train_eval is None or test_eval is None:
        raise RuntimeError("ZeroGraph 评估为空，请检查数据覆盖。")

    train_detail = train_eval["detail_df"].copy()
    test_detail = test_eval["detail_df"].copy()

    train_quarter = build_period_stats(train_detail, "Q", "zero")
    test_month = build_period_stats(test_detail, "M", "zero")

    summary_df = pd.DataFrame([{
        "model": EXP_NAME,
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
    }])

    train_detail.to_csv(os.path.join(RESULT_DIR, "predictions_ZeroGraph_train_detail.csv"), index=False, encoding="utf-8-sig")
    test_detail.to_csv(os.path.join(RESULT_DIR, "predictions_ZeroGraph_test_detail.csv"), index=False, encoding="utf-8-sig")
    train_eval["daily_df"].to_csv(os.path.join(RESULT_DIR, "daily_ZeroGraph_train.csv"), index=False, encoding="utf-8-sig")
    test_eval["daily_df"].to_csv(os.path.join(RESULT_DIR, "daily_ZeroGraph_test.csv"), index=False, encoding="utf-8-sig")
    train_quarter.to_csv(os.path.join(RESULT_DIR, "train_ZeroGraph_by_quarter.csv"), index=False, encoding="utf-8-sig")
    test_month.to_csv(os.path.join(RESULT_DIR, "test_ZeroGraph_by_month.csv"), index=False, encoding="utf-8-sig")
    summary_df.to_csv(os.path.join(RESULT_DIR, "summary_ZeroGraph.csv"), index=False, encoding="utf-8-sig")

    print("\n完成：ZeroGraph baseline（validation-select）已保存到", RESULT_DIR)
    print(summary_df)


if __name__ == "__main__":
    main()
