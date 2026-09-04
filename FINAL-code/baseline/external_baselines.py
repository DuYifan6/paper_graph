# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# 0) CONFIGURATION
# =============================================================================
ROOT = Path(r"D:\PythonProject\LASSO_FINAL")
FAIR_ROOT = ROOT / "AE-3" / "benchmark_fairness"
CACHE_NPZ = ROOT / "common_preprocess_cache" / "common_tensor_cache.npz"
META_PKL = ROOT / "common_preprocess_cache" / "common_tensor_meta.pkl"

OUT_DIR = FAIR_ROOT / "external"
RESULT_DIR = OUT_DIR / "results"
FLAT_DIR = OUT_DIR / "flat_cache_original_common"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FLAT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 20
SEED = 0
EXPECTED_FEATURES = 9

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

# Ridge
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]

# LightGBM
LGB_MAX_ROUNDS = 1200
LGB_PATIENCE = 50
LGB_LEARNING_RATE = 0.03
LGB_SUBSAMPLE = 0.80
LGB_SUBSAMPLE_FREQ = 1  # required for subsample<1 to take effect in LightGBM
LGB_COLSAMPLE_BYTREE = 0.80
LGB_GRID = [
    {"num_leaves": 31, "max_depth": -1, "min_child_samples": 20},
    {"num_leaves": 63, "max_depth": -1, "min_child_samples": 20},
    {"num_leaves": 31, "max_depth": 8, "min_child_samples": 50},
    {"num_leaves": 63, "max_depth": 8, "min_child_samples": 50},
]

# Transformer
TR_HIDDEN = 128
TR_NHEAD = 4
TR_LAYERS = 2
TR_DROPOUT = 0.10
TR_LR = 5e-4
TR_WEIGHT_DECAY = 1e-4
TR_EPOCHS = 20
TR_PATIENCE = 4
TR_BATCH_SIZE = 4096
TR_GRAD_CLIP = 1.0


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def target_days(start_idx: int, end_idx: int) -> np.ndarray:
    start = max(start_idx, LOOKBACK - 1)
    return np.arange(start, end_idx + 1, dtype=np.int32)


def validate_common_inputs(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    dates: List[pd.Timestamp],
    codes: List[str],
) -> None:
    """Fail closed if the shared tensor is not the paper's common panel."""
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


# =============================================================================
# 1) FLATTENED 20-DAY CACHE
# =============================================================================
def split_paths(name: str) -> Dict[str, Path]:
    return {
        "x": FLAT_DIR / f"{name}_X.dat",
        "y": FLAT_DIR / f"{name}_y.npy",
        "t": FLAT_DIR / f"{name}_t.npy",
        "n": FLAT_DIR / f"{name}_n.npy",
        "meta": FLAT_DIR / f"{name}_meta.json",
    }


def build_flat_split(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    days: np.ndarray,
) -> Tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray]:
    paths = split_paths(name)
    D = LOOKBACK * X.shape[2]
    n_rows = int(sum(int(mask[int(t)].sum()) for t in days))

    if all(paths[k].exists() for k in ["x", "y", "t", "n", "meta"]):
        with open(paths["meta"], "r", encoding="utf-8") as f:
            m = json.load(f)
        cached_days = m.get("target_day_indices")
        if (
            int(m.get("n_rows", -1)) == n_rows
            and int(m.get("dim", -1)) == D
            and int(m.get("lookback", -1)) == LOOKBACK
            and cached_days == [int(t) for t in days]
        ):
            Xmm = np.memmap(paths["x"], dtype="float32", mode="r", shape=(n_rows, D))
            return Xmm, np.load(paths["y"]), np.load(paths["t"]), np.load(paths["n"])

    print(f"[FLAT] building {name}: rows={n_rows:,}, dim={D}")
    Xmm = np.memmap(paths["x"], dtype="float32", mode="w+", shape=(n_rows, D))
    yy = np.empty(n_rows, dtype=np.float32)
    tt = np.empty(n_rows, dtype=np.int32)
    nn_idx = np.empty(n_rows, dtype=np.int32)

    pos = 0
    for t in map(int, days):
        idx = np.flatnonzero(mask[t])
        m = len(idx)
        if m == 0:
            continue
        # IMPORTANT: X[t] is already t-1 information. Do not exclude it.
        seq = np.asarray(X[t - LOOKBACK + 1:t + 1, idx, :], dtype=np.float32)
        block = np.transpose(seq, (1, 0, 2)).reshape(m, D)
        Xmm[pos:pos + m] = block
        yy[pos:pos + m] = np.asarray(y[t, idx], dtype=np.float32)
        tt[pos:pos + m] = t
        nn_idx[pos:pos + m] = idx.astype(np.int32)
        pos += m

    if pos != n_rows:
        raise RuntimeError(f"Flat-cache row mismatch: expected={n_rows}, wrote={pos}")
    Xmm.flush()
    np.save(paths["y"], yy)
    np.save(paths["t"], tt)
    np.save(paths["n"], nn_idx)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_rows": n_rows,
                "dim": D,
                "lookback": LOOKBACK,
                "target_day_indices": [int(t) for t in days],
            },
            f,
            indent=2,
        )
    return np.memmap(paths["x"], dtype="float32", mode="r", shape=(n_rows, D)), yy, tt, nn_idx


def prediction_df(y_true, y_pred, t_idx, n_idx, dates, codes, model_name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": [str(pd.Timestamp(dates[int(t)]).date()) for t in t_idx],
        "stock": np.asarray(codes, dtype=object)[n_idx].astype(str),
        "y_true": np.asarray(y_true, dtype=np.float32),
        "y_pred": np.asarray(y_pred, dtype=np.float32),
        "model": model_name,
    })


def basic_summary(model: str, val_y, val_pred, test_y, test_pred, extra: Dict) -> Dict:
    return {
        "model": model,
        "selection_criterion": "minimum validation MSE",
        "validation_mse": float(mean_squared_error(val_y, val_pred)),
        "validation_r2": float(r2_score(val_y, val_pred)),
        "test_mse": float(mean_squared_error(test_y, test_pred)),
        "test_r2": float(r2_score(test_y, test_pred)),
        "n_test": int(len(test_y)),
        **extra,
    }


# =============================================================================
# 2) RIDGE
# =============================================================================
def run_ridge(Xfit, yfit, Xval, yval, Xtest, ytest, ttest, ntest, dates, codes):
    records = []
    best = None
    for alpha in RIDGE_ALPHAS:
        t0 = time.time()
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(Xfit, yfit)
        vp = model.predict(Xval).astype(np.float32)
        vmse = float(mean_squared_error(yval, vp))
        records.append({"alpha": alpha, "val_mse": vmse, "elapsed_sec": time.time() - t0})
        print(f"[Ridge] alpha={alpha:g} val_mse={vmse:.8f}")
        if best is None or vmse < best[0]:
            best = (vmse, alpha, model)

    _, best_alpha, best_model = best
    test_pred = best_model.predict(Xtest).astype(np.float32)
    val_pred = best_model.predict(Xval).astype(np.float32)
    pd.DataFrame(records).to_csv(RESULT_DIR / "ridge_validation_grid.csv", index=False, encoding="utf-8-sig")
    pdf = prediction_df(ytest, test_pred, ttest, ntest, dates, codes, "Ridge_MSEFair")
    pdf.to_csv(RESULT_DIR / "pred_ridge_test.csv", index=False, encoding="utf-8-sig")
    return basic_summary(
        "Ridge_MSEFair", yval, val_pred, ytest, test_pred,
        {"selected_alpha": best_alpha, "training_loss": "squared error", "lookback": LOOKBACK}
    )


# =============================================================================
# 3) LIGHTGBM
# =============================================================================
def run_lightgbm(Xfit, yfit, Xval, yval, Xtest, ytest, ttest, ntest, dates, codes):
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("Install lightgbm first: pip install lightgbm") from e

    records = []
    best = None
    for cid, params in enumerate(LGB_GRID, 1):
        t0 = time.time()
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=LGB_MAX_ROUNDS,
            learning_rate=LGB_LEARNING_RATE,
            subsample=LGB_SUBSAMPLE,
            subsample_freq=LGB_SUBSAMPLE_FREQ,
            colsample_bytree=LGB_COLSAMPLE_BYTREE,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            **params,
        )
        model.fit(
            Xfit,
            yfit,
            eval_set=[(Xval, yval)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(LGB_PATIENCE, verbose=False), lgb.log_evaluation(period=50)],
        )
        best_iter = int(model.best_iteration_ or LGB_MAX_ROUNDS)
        vp = model.predict(Xval, num_iteration=best_iter).astype(np.float32)
        vmse = float(mean_squared_error(yval, vp))
        rec = {"candidate": cid, **params, "best_iteration": best_iter, "val_mse": vmse, "elapsed_sec": time.time() - t0}
        records.append(rec)
        print(f"[LightGBM] candidate={cid} best_iter={best_iter} val_mse={vmse:.8f} params={params}")
        if best is None or vmse < best[0]:
            best = (vmse, params.copy(), best_iter, model)

    _, best_params, best_iter, best_model = best
    val_pred = best_model.predict(Xval, num_iteration=best_iter).astype(np.float32)
    test_pred = best_model.predict(Xtest, num_iteration=best_iter).astype(np.float32)
    pd.DataFrame(records).to_csv(RESULT_DIR / "lightgbm_validation_grid.csv", index=False, encoding="utf-8-sig")
    pdf = prediction_df(ytest, test_pred, ttest, ntest, dates, codes, "LightGBM_MSEFair")
    pdf.to_csv(RESULT_DIR / "pred_lightgbm_test.csv", index=False, encoding="utf-8-sig")
    return basic_summary(
        "LightGBM_MSEFair", yval, val_pred, ytest, test_pred,
        {
            "training_loss": "regression MSE",
            "selected_num_leaves": best_params["num_leaves"],
            "selected_max_depth": best_params["max_depth"],
            "selected_min_child_samples": best_params["min_child_samples"],
            "selected_boosting_round": best_iter,
            "max_boosting_rounds": LGB_MAX_ROUNDS,
            "early_stopping_patience": LGB_PATIENCE,
            "subsample": LGB_SUBSAMPLE,
            "subsample_freq": LGB_SUBSAMPLE_FREQ,
            "colsample_bytree": LGB_COLSAMPLE_BYTREE,
            "fit_row_cap": 0,
            "lookback": LOOKBACK,
        },
    )


# =============================================================================
# 4) TRANSFORMER
# =============================================================================
class FlatMemmapDataset(Dataset):
    def __init__(self, Xmm: np.memmap, y: np.ndarray, L: int, F: int):
        self.X = Xmm
        self.y = y
        self.L = L
        self.F = F

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx], dtype=np.float32).reshape(self.L, self.F).copy()
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.float32)


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
        return x + self.pe[:, :x.size(1)]


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, TR_HIDDEN)
        self.pe = PositionalEncoding(TR_HIDDEN)
        layer = nn.TransformerEncoderLayer(
            d_model=TR_HIDDEN,
            nhead=TR_NHEAD,
            dim_feedforward=TR_HIDDEN * 4,
            dropout=TR_DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=TR_LAYERS)
        self.head = nn.Sequential(nn.LayerNorm(TR_HIDDEN), nn.Linear(TR_HIDDEN, 1))

    def forward(self, x):
        z = self.encoder(self.pe(self.proj(x)))
        return self.head(z[:, -1, :]).squeeze(-1)


def evaluate_transformer(model, loader, device) -> Tuple[float, np.ndarray]:
    model.eval()
    sse, n, preds = 0.0, 0, []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            p = model(xb)
            e = p - yb
            sse += float(torch.sum(e * e).item())
            n += int(yb.numel())
            preds.append(p.detach().cpu().numpy())
    return (sse / n if n else float("inf")), np.concatenate(preds).astype(np.float32)


def run_transformer(Xfit, yfit, Xval, yval, Xtest, ytest, ttest, ntest, dates, codes, F: int):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerRegressor(F).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TR_LR, weight_decay=TR_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    fit_loader = DataLoader(FlatMemmapDataset(Xfit, yfit, LOOKBACK, F), batch_size=TR_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(FlatMemmapDataset(Xval, yval, LOOKBACK, F), batch_size=TR_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(FlatMemmapDataset(Xtest, ytest, LOOKBACK, F), batch_size=TR_BATCH_SIZE, shuffle=False, num_workers=0)

    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0
    hist = []
    for ep in range(1, TR_EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_sse, train_n = 0.0, 0
        for xb, yb in fit_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if TR_GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), TR_GRAD_CLIP)
            opt.step()
            train_sse += float(torch.sum((pred.detach() - yb) ** 2).item())
            train_n += int(yb.numel())

        train_mse = train_sse / max(train_n, 1)
        val_mse, _ = evaluate_transformer(model, val_loader, device)
        hist.append({"epoch": ep, "train_mse": train_mse, "val_mse": val_mse, "elapsed_sec": time.time() - t0})
        print(f"[Transformer] epoch={ep:02d}/{TR_EPOCHS} train_mse={train_mse:.8f} val_mse={val_mse:.8f}")

        if val_mse < best_val - 1e-8:
            best_val = val_mse
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= TR_PATIENCE:
                print(f"[Transformer] early stop at epoch={ep}; best_epoch={best_epoch}")
                break

    if best_state is None:
        raise RuntimeError("Transformer selected no checkpoint.")
    model.load_state_dict(best_state)
    torch.save(best_state, RESULT_DIR / "best_transformer.pt")
    pd.DataFrame(hist).to_csv(RESULT_DIR / "transformer_training_history.csv", index=False, encoding="utf-8-sig")

    val_mse, val_pred = evaluate_transformer(model, val_loader, device)
    test_mse, test_pred = evaluate_transformer(model, test_loader, device)
    pdf = prediction_df(ytest, test_pred, ttest, ntest, dates, codes, "Transformer_MSEFair")
    pdf.to_csv(RESULT_DIR / "pred_transformer_test.csv", index=False, encoding="utf-8-sig")
    return basic_summary(
        "Transformer_MSEFair", yval, val_pred, ytest, test_pred,
        {
            "training_loss": "MSE",
            "best_epoch": best_epoch,
            "max_epochs": TR_EPOCHS,
            "early_stopping_patience": TR_PATIENCE,
            "hidden": TR_HIDDEN,
            "layers": TR_LAYERS,
            "nhead": TR_NHEAD,
            "lookback": LOOKBACK,
            "device": str(device),
        },
    )


# =============================================================================
# 5) MAIN
# =============================================================================
def main():
    set_seed(SEED)
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
    fit_days = target_days(train_start, fit_end)
    val_days = target_days(val_start, train_end)
    test_days = target_days(test_start, test_end)
    validate_paper_split(
        dates, fit_end, val_start, train_end, test_start, test_end, test_days
    )

    print(f"[SPLIT] fit targets: {dates[fit_days[0]].date()} .. {dates[fit_days[-1]].date()}")
    print(f"[SPLIT] validation: {dates[val_days[0]].date()} .. {dates[val_days[-1]].date()}")
    print(f"[SPLIT] test: {dates[test_days[0]].date()} .. {dates[test_days[-1]].date()}")

    Xfit, yfit, _, _ = build_flat_split("fit", X, y, mask, fit_days)
    Xval, yval, _, _ = build_flat_split("validation", X, y, mask, val_days)
    Xtest, ytest, ttest, ntest = build_flat_split("test", X, y, mask, test_days)

    print(f"[DATA] fit={Xfit.shape}, validation={Xval.shape}, test={Xtest.shape}")
    summaries = []
    summaries.append(run_ridge(Xfit, yfit, Xval, yval, Xtest, ytest, ttest, ntest, dates, codes))
    summaries.append(run_lightgbm(Xfit, yfit, Xval, yval, Xtest, ytest, ttest, ntest, dates, codes))
    summaries.append(run_transformer(Xfit, yfit, Xval, yval, Xtest, ytest, ttest, ntest, dates, codes, F=X.shape[2]))

    out = pd.DataFrame(summaries)
    out.to_csv(RESULT_DIR / "summary_external_msefair.csv", index=False, encoding="utf-8-sig")
    print("\nFINAL EXTERNAL FAIRNESS SUMMARY")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
