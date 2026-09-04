# -*- coding: utf-8 -*-
"""
Hybrid1455_BuildLatestTensor_FINAL.py
=====================================

FINAL builder after diagnosing the TRUE common-cache structure.

Confirmed cache mapping from the diagnostic:
------------------------------------------------
cache dim 0 -> D_bull_bear_ratio
cache dim 1 -> D_intraday_return
cache dim 2 -> D_mainforce_strength
cache dim 3 -> D_overnight_return
cache dim 4 -> D_pct_chg
cache dim 5 -> M_MACD_1
cache dim 6 -> M_MACD_hist_1
cache dim 7 -> M_RSI_6_5
cache dim 8 -> M_high_1

Confirmed global date offset:
    cache X[t] aligns with source feature date all_dates[t-1].

Therefore, for forecasting target day t:
    X[t] = original t-1 feature vector.

Hybrid1455 latest-step protocol
-------------------------------
Keep original X[t] for all features except:

1) D_pct_chg (dim 4):
       reconstruct at 14:55 on t-1,
       then map DIRECTLY into the ACTUAL CACHE SCALE
       using train-only (2022-2024) Huber calibration.

2) D_intraday_return (dim 1):
       same as above.

3) D_mainforce_strength (dim 2):
       use X[t-1, dim 2], which corresponds to t-2.

4) D_bull_bear_ratio (dim 0):
       use X[t-1, dim 0], which corresponds to t-2.

All other dimensions remain exactly as stored in X[t].

Important improvement
---------------------
The calibration target is no longer the external daily CSV.
It is the ACTUAL common-cache value used by the forecasting model:

    raw full-close return on signal date d
        -> X[t, stock, cache_dim], where d = all_dates[t-1]

This guarantees the mapped 14:55 value is on the model's true input scale.

NO 2025 observation is used to fit slopes/intercepts or clipping bounds.
"""

import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score


# =========================================================
# 0) CONFIG
# =========================================================

COMMON_CACHE_NPZ = (
    r"D:\PythonProject\LASSO_FINAL\common_preprocess_cache"
    r"\common_tensor_cache.npz"
)
META_PKL = (
    r"D:\PythonProject\LASSO_FINAL\common_preprocess_cache"
    r"\common_tensor_meta.pkl"
)

MIN5_DIR = r"D:\Lasso\min5_aligned"

OUT_DIR = (
    r"D:\PythonProject\LASSO_FINAL\graph_backtest"
    r"\economic_executable\hybrid1455"
)
CACHE_DIR = os.path.join(OUT_DIR, "cache")
AUDIT_DIR = os.path.join(OUT_DIR, "audit_final")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AUDIT_DIR, exist_ok=True)

OUT_TENSOR_NPZ = os.path.join(
    CACHE_DIR, "hybrid1455_latest_tensor.npz"
)
OUT_META_JSON = os.path.join(
    CACHE_DIR, "hybrid1455_latest_tensor_meta.json"
)

SANITY_ONLY = False
SANITY_N_STOCKS = 20
RANDOM_SEED = 20260820

TRAIN_START = "2022-01-01"
TRAIN_END = "2024-12-31"

T_0930 = "09:30"
T_1455 = "14:55"
T_1500 = "15:00"

HUBER_EPSILON = 1.35
HUBER_ALPHA = 0.0
MIN_CALIBRATION_OBS = 100

# Fixed audit gates, not forecasting hyperparameters.
MIN_MEDIAN_INLIER_R2 = 0.98
MAX_ALLOWED_ERRORS_SANITY = 0

# Robust residual rule used only to evaluate mapping fidelity.
OUTLIER_MAD_MULT = 4.5


# =========================================================
# 1) CONFIRMED CACHE STRUCTURE
# =========================================================

CACHE_DIM = {
    "D_bull_bear_ratio": 0,
    "D_intraday_return": 1,
    "D_mainforce_strength": 2,
    "D_overnight_return": 3,
    "D_pct_chg": 4,
    "M_MACD_1": 5,
    "M_MACD_hist_1": 6,
    "M_RSI_6_5": 7,
    "M_high_1": 8,
}

CACHE_DATE_OFFSET = -1

DIM_PCT = CACHE_DIM["D_pct_chg"]
DIM_INTRA = CACHE_DIM["D_intraday_return"]
DIM_MAINFORCE = CACHE_DIM["D_mainforce_strength"]
DIM_BULLBEAR = CACHE_DIM["D_bull_bear_ratio"]


# =========================================================
# 2) HELPERS
# =========================================================

def safe_read_csv(fp, **kwargs):
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312"]:
        try:
            return pd.read_csv(
                fp, encoding=enc, low_memory=False, **kwargs
            )
        except Exception as e:
            last_err = e
    raise last_err


def normalize_code(x):
    return str(x).strip().upper()


def code_from_filename(fp):
    base = os.path.basename(fp)
    first = base.split("_")[0].replace(".csv", "")
    return normalize_code(first)


def build_file_map(folder):
    out = {}
    for fp in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        code = code_from_filename(fp)
        if (
            code not in out
            or len(os.path.basename(fp))
            < len(os.path.basename(out[code]))
        ):
            out[code] = fp
    return out


def load_min5(fp):
    d = safe_read_csv(fp)

    required = ["trade_time", "open", "close"]
    miss = [c for c in required if c not in d.columns]
    if miss:
        raise ValueError(f"5min missing {miss}: {fp}")

    d["trade_time"] = pd.to_datetime(
        d["trade_time"], errors="coerce"
    )
    d = d.dropna(subset=["trade_time"]).copy()

    d["date"] = d["trade_time"].dt.normalize()
    d["clock"] = d["trade_time"].dt.strftime("%H:%M")

    d["open"] = pd.to_numeric(d["open"], errors="coerce")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")

    return d.sort_values("trade_time")


# =========================================================
# 3) RAW PRICE / RETURN TABLE
# =========================================================

def make_price_table(m5):
    def point(clock, col, out_name):
        q = m5[m5["clock"] == clock][
            ["date", col]
        ].copy()
        q = q.drop_duplicates("date", keep="last")
        return q.rename(columns={col: out_name})

    p0930 = point(T_0930, "open", "open_0930")
    p1455 = point(T_1455, "close", "close_1455")
    p1500 = point(T_1500, "close", "close_1500")

    p = p0930.merge(p1455, on="date", how="outer")
    p = p.merge(p1500, on="date", how="outer")
    p = p.sort_values("date").reset_index(drop=True)

    # Previous observed full-day close for this stock.
    p["prev_close_1500"] = p["close_1500"].shift(1)

    # Full-close raw features, used ONLY to learn cache scale.
    p["pct_full_raw_pct"] = (
        (p["close_1500"] / p["prev_close_1500"] - 1.0)
        * 100.0
    )
    p["intra_full_raw_pct"] = (
        (p["close_1500"] / p["open_0930"] - 1.0)
        * 100.0
    )

    # Executable 14:55 raw versions.
    p["pct_1455_raw_pct"] = (
        (p["close_1455"] / p["prev_close_1500"] - 1.0)
        * 100.0
    )
    p["intra_1455_raw_pct"] = (
        (p["close_1455"] / p["open_0930"] - 1.0)
        * 100.0
    )

    return p


# =========================================================
# 4) ROBUST CALIBRATION DIRECTLY TO CACHE SCALE
# =========================================================

def robust_outlier_mask(residual):
    r = np.asarray(residual, dtype=float)

    center = np.median(r)
    mad = np.median(np.abs(r - center))
    sigma = 1.4826 * mad

    if not np.isfinite(sigma) or sigma < 1e-12:
        return np.zeros(len(r), dtype=bool)

    return (
        np.abs(r - center)
        > OUTLIER_MAD_MULT * sigma
    )


def fit_huber_cache_scale(raw_x, cache_y):
    q = pd.DataFrame({
        "x": pd.to_numeric(raw_x, errors="coerce"),
        "y": pd.to_numeric(cache_y, errors="coerce"),
    }).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(q) < MIN_CALIBRATION_OBS:
        return None

    X = q["x"].to_numpy(dtype=float).reshape(-1, 1)
    y = q["y"].to_numpy(dtype=float)

    model = HuberRegressor(
        epsilon=HUBER_EPSILON,
        alpha=HUBER_ALPHA,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-8,
    )
    model.fit(X, y)

    pred = model.predict(X)
    resid = y - pred
    outlier = robust_outlier_mask(resid)

    r2_all = (
        float(r2_score(y, pred))
        if len(np.unique(y)) > 1 else np.nan
    )

    keep = ~outlier
    if keep.sum() >= 2 and len(np.unique(y[keep])) > 1:
        r2_inlier = float(
            r2_score(y[keep], pred[keep])
        )
    else:
        r2_inlier = np.nan

    return {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2_all": r2_all,
        "r2_inlier": r2_inlier,
        "n_train": int(len(q)),
        "n_outlier": int(outlier.sum()),
        "outlier_rate": float(outlier.mean()),
        # IMPORTANT: clipping bounds are on ACTUAL CACHE values.
        "clip_lo": float(np.min(y)),
        "clip_hi": float(np.max(y)),
    }


def build_calibration_frame(
    prices,
    X_np,
    n_idx,
    all_dates,
    dim,
    raw_full_col,
    raw_1455_col,
):
    """
    Confirmed alignment:
        X_np[t, n, dim] corresponds to source feature date all_dates[t-1].

    Therefore for each signal/source date d=all_dates[t-1],
    pair raw return on d with cache value X[t].
    """
    date_to_price = prices.set_index("date")
    rows = []

    for t in range(1, len(all_dates)):
        signal_date = all_dates[t - 1]

        if signal_date not in date_to_price.index:
            continue

        r = date_to_price.loc[signal_date]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[-1]

        raw_full = r[raw_full_col]
        raw_1455 = r[raw_1455_col]
        cache_val = X_np[t, n_idx, dim]

        rows.append({
            "t": t,
            "signal_date": signal_date,
            "raw_full": raw_full,
            "raw_1455": raw_1455,
            "cache_value": cache_val,
        })

    return pd.DataFrame(rows)


# =========================================================
# 5) STOCK PROCESSING
# =========================================================

def process_stock(
    code,
    n_idx,
    min5_fp,
    X_np,
    H_np,
    hybrid_mask,
    all_dates,
):
    m5 = load_min5(min5_fp)
    prices = make_price_table(m5)

    # ---------------- D_pct_chg ----------------
    pct_frame = build_calibration_frame(
        prices=prices,
        X_np=X_np,
        n_idx=n_idx,
        all_dates=all_dates,
        dim=DIM_PCT,
        raw_full_col="pct_full_raw_pct",
        raw_1455_col="pct_1455_raw_pct",
    )

    pct_train = pct_frame[
        (pct_frame["signal_date"] >= pd.Timestamp(TRAIN_START))
        & (pct_frame["signal_date"] <= pd.Timestamp(TRAIN_END))
    ].copy()

    p_pct = fit_huber_cache_scale(
        pct_train["raw_full"],
        pct_train["cache_value"],
    )
    if p_pct is None:
        raise RuntimeError("D_pct_chg calibration insufficient")

    # ---------------- D_intraday_return ----------------
    intra_frame = build_calibration_frame(
        prices=prices,
        X_np=X_np,
        n_idx=n_idx,
        all_dates=all_dates,
        dim=DIM_INTRA,
        raw_full_col="intra_full_raw_pct",
        raw_1455_col="intra_1455_raw_pct",
    )

    intra_train = intra_frame[
        (intra_frame["signal_date"] >= pd.Timestamp(TRAIN_START))
        & (intra_frame["signal_date"] <= pd.Timestamp(TRAIN_END))
    ].copy()

    p_intra = fit_huber_cache_scale(
        intra_train["raw_full"],
        intra_train["cache_value"],
    )
    if p_intra is None:
        raise RuntimeError(
            "D_intraday_return calibration insufficient"
        )

    # ---------------- fill 14:55 cache-scale values ----------------
    filled_pct = 0
    for _, r in pct_frame.iterrows():
        t = int(r["t"])
        raw = r["raw_1455"]
        if not np.isfinite(raw):
            continue

        z = (
            p_pct["slope"] * float(raw)
            + p_pct["intercept"]
        )
        z = float(np.clip(
            z,
            p_pct["clip_lo"],
            p_pct["clip_hi"],
        ))

        H_np[t, n_idx, DIM_PCT] = z
        filled_pct += 1

    filled_intra = 0
    for _, r in intra_frame.iterrows():
        t = int(r["t"])
        raw = r["raw_1455"]
        if not np.isfinite(raw):
            continue

        z = (
            p_intra["slope"] * float(raw)
            + p_intra["intercept"]
        )
        z = float(np.clip(
            z,
            p_intra["clip_lo"],
            p_intra["clip_hi"],
        ))

        H_np[t, n_idx, DIM_INTRA] = z
        filled_intra += 1

    # Final eligibility for this stock.
    finite_latest = np.isfinite(
        H_np[:, n_idx, :]
    ).all(axis=1)
    hybrid_mask[:, n_idx] &= finite_latest

    calib_row = {
        "stock": code,

        "pct_slope": p_pct["slope"],
        "pct_intercept": p_pct["intercept"],
        "pct_r2_all": p_pct["r2_all"],
        "pct_r2_inlier": p_pct["r2_inlier"],
        "pct_n_train": p_pct["n_train"],
        "pct_outlier_rate": p_pct["outlier_rate"],
        "pct_clip_lo": p_pct["clip_lo"],
        "pct_clip_hi": p_pct["clip_hi"],

        "intra_slope": p_intra["slope"],
        "intra_intercept": p_intra["intercept"],
        "intra_r2_all": p_intra["r2_all"],
        "intra_r2_inlier": p_intra["r2_inlier"],
        "intra_n_train": p_intra["n_train"],
        "intra_outlier_rate": p_intra["outlier_rate"],
        "intra_clip_lo": p_intra["clip_lo"],
        "intra_clip_hi": p_intra["clip_hi"],

        "filled_pct_days": filled_pct,
        "filled_intra_days": filled_intra,
        "hybrid_valid_days": int(
            hybrid_mask[:, n_idx].sum()
        ),
    }

    return calib_row


# =========================================================
# 6) MAIN
# =========================================================

def main():
    print("=== Hybrid1455 FINAL latest-step tensor builder ===")
    print("[SANITY_ONLY] =", SANITY_ONLY)
    print("[CONFIRMED CACHE DATE OFFSET] =", CACHE_DATE_OFFSET)
    print("[CONFIRMED CACHE DIM MAP]")
    for k, v in sorted(
        CACHE_DIM.items(), key=lambda x: x[1]
    ):
        print(f"  dim {v}: {k}")

    pack = np.load(COMMON_CACHE_NPZ)
    X_np = pack["X_np"].astype(np.float32)
    y_np = pack["y_np"].astype(np.float32)
    original_mask = pack["mask_np"].astype(bool)

    with open(META_PKL, "rb") as f:
        meta = pickle.load(f)

    all_dates = [
        pd.Timestamp(d).normalize()
        for d in meta["all_dates"]
    ]
    codes = [
        normalize_code(c)
        for c in meta["ts_codes_order"]
    ]

    T, N, F = X_np.shape
    print(f"[CACHE] T={T}, N={N}, F={F}")

    if F != 9:
        raise RuntimeError(
            f"Expected 9 cache dimensions, found {F}"
        )

    min5_map = build_file_map(MIN5_DIR)
    code_to_n = {c: i for i, c in enumerate(codes)}
    available_codes = [
        c for c in codes if c in min5_map
    ]

    print(
        "[FILES] min5 available for cache universe =",
        len(available_codes),
        "/",
        len(codes),
    )

    if SANITY_ONLY:
        rng = np.random.default_rng(RANDOM_SEED)
        if len(available_codes) > SANITY_N_STOCKS:
            process_codes = sorted(
                rng.choice(
                    available_codes,
                    size=SANITY_N_STOCKS,
                    replace=False,
                ).tolist()
            )
        else:
            process_codes = available_codes
    else:
        process_codes = available_codes

    # Start from actual cache values.
    H_np = X_np.copy()
    hybrid_mask = original_mask.copy()

    # Conservative t-2 replacement for the two supplied fund-flow features.
    # Confirmed cache alignment:
    #   X[t]   -> source day t-1
    #   X[t-1] -> source day t-2
    H_np[1:, :, DIM_MAINFORCE] = (
        X_np[:-1, :, DIM_MAINFORCE]
    )
    H_np[1:, :, DIM_BULLBEAR] = (
        X_np[:-1, :, DIM_BULLBEAR]
    )

    H_np[0, :, DIM_MAINFORCE] = np.nan
    H_np[0, :, DIM_BULLBEAR] = np.nan
    hybrid_mask[0, :] = False

    # Never allow accidental fallback to full-close values
    # for the two features that must be reconstructed at 14:55.
    H_np[:, :, DIM_PCT] = np.nan
    H_np[:, :, DIM_INTRA] = np.nan

    calib_rows = []
    error_rows = []

    for code in tqdm(
        process_codes,
        desc="Hybrid1455 cache-scale calibration/build",
    ):
        n_idx = code_to_n[code]

        try:
            row = process_stock(
                code=code,
                n_idx=n_idx,
                min5_fp=min5_map[code],
                X_np=X_np,
                H_np=H_np,
                hybrid_mask=hybrid_mask,
                all_dates=all_dates,
            )
            calib_rows.append(row)

        except Exception as e:
            hybrid_mask[:, n_idx] = False
            error_rows.append({
                "stock": code,
                "error": repr(e),
            })

    calib_df = pd.DataFrame(calib_rows)
    errors_df = pd.DataFrame(error_rows)

    calib_path = os.path.join(
        AUDIT_DIR,
        "hybrid1455_cache_scale_calibration.csv",
    )
    errors_path = os.path.join(
        AUDIT_DIR,
        "hybrid1455_cache_scale_errors.csv",
    )

    calib_df.to_csv(
        calib_path,
        index=False,
        encoding="utf-8-sig",
    )
    errors_df.to_csv(
        errors_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== CACHE-SCALE CALIBRATION ==========")
    if not calib_df.empty:
        summary = calib_df[
            [
                "pct_r2_all",
                "pct_r2_inlier",
                "intra_r2_all",
                "intra_r2_inlier",
                "pct_outlier_rate",
                "intra_outlier_rate",
                "hybrid_valid_days",
            ]
        ].describe()

        print(summary.to_string())

        print("\nMedian inlier R2:")
        print(
            "D_pct_chg =",
            float(calib_df["pct_r2_inlier"].median()),
        )
        print(
            "D_intraday_return =",
            float(calib_df["intra_r2_inlier"].median()),
        )

    print("\n[ERRORS] =", len(errors_df))
    if len(errors_df):
        print(errors_df.head(30).to_string(index=False))

    # -----------------------------------------------------
    # SANITY gate
    # -----------------------------------------------------
    if SANITY_ONLY:
        if calib_df.empty:
            raise RuntimeError(
                "No successful calibration rows."
            )

        med_pct = float(
            calib_df["pct_r2_inlier"].median()
        )
        med_intra = float(
            calib_df["intra_r2_inlier"].median()
        )

        if (
            med_pct < MIN_MEDIAN_INLIER_R2
            or med_intra < MIN_MEDIAN_INLIER_R2
        ):
            raise RuntimeError(
                "Cache-scale calibration fidelity is too low: "
                f"pct={med_pct:.6f}, intra={med_intra:.6f}"
            )

        if len(errors_df) > MAX_ALLOWED_ERRORS_SANITY:
            raise RuntimeError(
                f"Sanity has {len(errors_df)} processing errors."
            )

        print("\n[SANITY PASS]")
        print(
            "Cache mapping/date alignment is now incorporated directly."
        )
        print(
            "Set SANITY_ONLY=False and run this SAME script "
            "to generate the formal full-market Hybrid1455 tensor."
        )
        print(
            "Formal tensor is NOT saved in sanity mode."
        )
        return

    # -----------------------------------------------------
    # Formal full-universe invalidation for missing 5m stocks
    # -----------------------------------------------------
    processed_set = set(process_codes)
    for code in codes:
        if code not in processed_set:
            hybrid_mask[:, code_to_n[code]] = False

    # Final finite requirement.
    finite_all = np.isfinite(H_np).all(axis=2)
    hybrid_mask &= finite_all

    H_save = np.nan_to_num(
        H_np,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    np.savez_compressed(
        OUT_TENSOR_NPZ,
        X_latest_hybrid_np=H_save,
        mask_hybrid_np=hybrid_mask.astype(np.uint8),
        cache_dim_names=np.asarray(
            [
                "D_bull_bear_ratio",
                "D_intraday_return",
                "D_mainforce_strength",
                "D_overnight_return",
                "D_pct_chg",
                "M_MACD_1",
                "M_MACD_hist_1",
                "M_RSI_6_5",
                "M_high_1",
            ],
            dtype=object,
        ),
    )

    metadata = {
        "version": "Hybrid1455_FINAL_cache_scale",
        "information_cutoff": "14:55",
        "execution": (
            "14:55 signal -> 14:57-15:00 closing auction "
            "-> next open"
        ),
        "confirmed_cache_date_offset": CACHE_DATE_OFFSET,
        "confirmed_cache_dim_mapping": CACHE_DIM,
        "calibration_target": (
            "actual common-cache dimension value"
        ),
        "calibration_train_start": TRAIN_START,
        "calibration_train_end": TRAIN_END,
        "uses_2025_for_calibration": False,
        "calibration_method": {
            "model": "HuberRegressor",
            "epsilon": HUBER_EPSILON,
            "alpha": HUBER_ALPHA,
            "clipping": (
                "per-stock actual-cache training min/max"
            ),
        },
        "hybrid_protocol": {
            "D_bull_bear_ratio": "t-2 cache value",
            "D_intraday_return": (
                "t-1 14:55 raw -> cache-scale mapping"
            ),
            "D_mainforce_strength": "t-2 cache value",
            "D_overnight_return": "original t-1 cache value",
            "D_pct_chg": (
                "t-1 14:55 raw -> cache-scale mapping"
            ),
            "M_MACD_1": "original t-1 cache value",
            "M_MACD_hist_1": "original t-1 cache value",
            "M_RSI_6_5": "original t-1 cache value",
            "M_high_1": "original t-1 cache value",
        },
        "tensor_shape": list(H_save.shape),
        "processed_stocks": len(process_codes),
        "errors": len(errors_df),
        "original_valid_obs": int(original_mask.sum()),
        "hybrid_valid_obs": int(hybrid_mask.sum()),
    }

    with open(
        OUT_META_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n========== FORMAL TENSOR SAVED ==========")
    print(OUT_TENSOR_NPZ)
    print(OUT_META_JSON)
    print(
        "[VALID OBS] original =",
        int(original_mask.sum()),
        "| hybrid =",
        int(hybrid_mask.sum()),
    )
    print(
        "[NEXT] Run Hybrid1455_ZeroGraph.py, "
        "then Hybrid1455_PositiveGraph2.py."
    )


if __name__ == "__main__":
    main()
