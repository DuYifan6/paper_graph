# -*- coding: utf-8 -*-
"""
Hybrid1455_AE_CLEAN_CostAware_Backtest.py
=========================================

AE-clean economic backtest for the Hybrid14:55 design.

Purpose
-------
This script fixes two issues in the earlier closing-auction backtest without
retraining either forecasting model and without using 2025 outcomes to choose
trading rules:

1) Information timing:
   ranking / participation uses information available by 14:55 only.
   15:00 auction information is used only to determine whether a submitted
   order is filled. If an order is not fillable, that slot remains cash and
   is NOT replaced ex post.

2) Forced full-K dilution:
   K is interpreted as the MAXIMUM number of positions. A stock is submitted
   only when its PRE-2025 calibrated expected overnight return exceeds a fixed
   conservative all-in execution hurdle. Unused slots remain cash.

Frozen deployment timeline
--------------------------
day t-1:
    14:55   information cutoff / candidate screen / ranking / participation
    14:57-15:00 submit closing-auction orders
    15:00   official close used as execution-price proxy for filled orders

day t:
    09:30   exit at next-day open (sell-limit risk is flagged, not deleted)

Calibration (NO 2025 tuning)
----------------------------
- Calibration sample: chronological last 20% of PRE-2025 prediction dates.
- One monotonic calibration map is estimated separately for Graph-2 and
  ZeroGraph using IsotonicRegression(y_pred -> realized overnight return).
- Participation hurdle is fixed at the exact break-even gross return under:
      buy cost  = 0.01%
      sell cost = 0.11%
      + 10 bp slippage PER SIDE
  Thus the activation rule is deliberately more conservative than the
  baseline 0-bp slippage backtest.
- No 2025 return, no 2025 K result, and no 2025 threshold search is used.

Portfolio rule
--------------
For K in {1,3,5,10}:
- select at most K stocks whose calibrated expected gross overnight return
  exceeds the frozen hurdle;
- each potential slot has capital weight 1/K;
- fewer than K signals => remaining slots stay in cash;
- an order that fails the 15:00 fill-feasibility check stays cash;
- no post-14:55 replacement is allowed.

Required existing inputs
------------------------
1) Graph-2 Hybrid1455 PRE-2025 train detail predictions
2) Graph-2 Hybrid1455 2025 test detail predictions
3) ZeroGraph Hybrid1455 PRE-2025 train detail predictions
4) ZeroGraph Hybrid1455 2025 test detail predictions
5) market_execution_cache.csv created by the previous closing-auction audit

Outputs
-------
calibration_summary.csv
calibration_tail_diagnostics.csv
summary_costaware.csv
daily_returns_costaware.csv
selected_signals_costaware.csv
real_zero_daily_diff_hac_costaware.csv
goal_diagnostic_only.csv

IMPORTANT
---------
The goal diagnostic reports whether realized 2025 performance is positive and
whether K=10 exceeds the old-paper 48.38% reference. It is diagnostic ONLY and
never feeds back into calibration or trading-rule selection.
"""

import os
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

# =========================================================
# 0) PATHS
# =========================================================

BASE = r"D:\PythonProject\LASSO_FINAL\graph_backtest\economic_executable\hybrid1455"

REAL_DIR = os.path.join(BASE, "graph2_positive", "results")
ZERO_DIR = os.path.join(BASE, "zerograph_matched", "results")
OLD_BT_DIR = os.path.join(BASE, "closing_auction_backtest")
OUT_DIR = os.path.join(BASE, "closing_auction_costaware_ae_clean")
os.makedirs(OUT_DIR, exist_ok=True)

# Candidate filenames are resolved in order so the script tolerates the two
# naming conventions that appeared in the project.
REAL_TRAIN_CANDIDATES = [
    os.path.join(REAL_DIR, "predictions_Graph2_PositiveOnly_Hybrid1455_C_K3510_train_detail.csv"),
]
REAL_TEST_CANDIDATES = [
    os.path.join(REAL_DIR, "predictions_Graph2_PositiveOnly_Hybrid1455_C_K3510_test_detail.csv"),
]
ZERO_TRAIN_CANDIDATES = [
    os.path.join(ZERO_DIR, "predictions_ZeroGraph_Matched_Hybrid1455_train_detail.csv"),
    os.path.join(ZERO_DIR, "predictions_ZeroGraph_Matched_train_detail.csv"),
]
ZERO_TEST_CANDIDATES = [
    os.path.join(ZERO_DIR, "predictions_ZeroGraph_Matched_Hybrid1455_test_detail.csv"),
    os.path.join(ZERO_DIR, "predictions_ZeroGraph_Matched_test_detail.csv"),
]

MARKET_CACHE_CSV = os.path.join(OLD_BT_DIR, "market_execution_cache.csv")

# =========================================================
# 1) FROZEN RULES
# =========================================================

TOPK_LIST = [1, 3, 5, 10]
SLIPPAGE_BPS_LIST = [0, 5, 10]  # per side, only for reported sensitivity

BUY_COST = 0.0001
SELL_COST = 0.0011
CALIBRATION_SLIPPAGE_BPS_PER_SIDE = 10  # frozen conservative activation hurdle
CALIBRATION_RATIO = 0.20               # last 20% PRE-2025 dates
MIN_CALIBRATION_DATES = 20

BUY_LIMIT_THRESHOLD = 0.09
SELL_LIMIT_RISK_THRESHOLD = -0.09
HAC_LAG = 2
TRADING_DAYS_PER_YEAR = 252
Y_TRUE_IS_PERCENT = True

# Old-paper reference is diagnostic only; never used by the strategy.
OLD_PAPER_K10_TOTAL_RETURN = 0.4838

# =========================================================
# 2) HELPERS
# =========================================================

def safe_read_csv(fp):
    last = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312"]:
        try:
            return pd.read_csv(fp, encoding=enc, low_memory=False)
        except Exception as e:
            last = e
    raise last


def resolve_existing(candidates, label):
    for fp in candidates:
        if os.path.exists(fp):
            print(f"[PATH] {label}: {fp}")
            return fp
    raise FileNotFoundError(
        f"Cannot find {label}. Tried:\n" + "\n".join(candidates)
    )


def normalize_stock_series(s):
    return s.astype(str).str.strip().str.upper()


def exact_net_return(gross_ret, slip_bps):
    slip = float(slip_bps) / 10000.0
    buy_den = (1.0 + BUY_COST) * (1.0 + slip)
    sell_mult = 1.0 - SELL_COST - slip
    if sell_mult <= 0:
        raise ValueError("sell cost + slippage invalid")
    return (1.0 + np.asarray(gross_ret, dtype=float)) * sell_mult / buy_den - 1.0


def exact_break_even_gross(slip_bps):
    slip = float(slip_bps) / 10000.0
    buy_den = (1.0 + BUY_COST) * (1.0 + slip)
    sell_mult = 1.0 - SELL_COST - slip
    return float(buy_den / sell_mult - 1.0)


def max_drawdown(equity):
    e = np.asarray(equity, dtype=float)
    if len(e) == 0:
        return np.nan
    peak = np.maximum.accumulate(e)
    return float(np.min(e / np.maximum(peak, 1e-18) - 1.0))


def sharpe_zero_rf(r):
    r = np.asarray(r, dtype=float)
    if len(r) <= 1:
        return np.nan
    sd = np.std(r, ddof=1)
    return float(np.mean(r) / sd * np.sqrt(TRADING_DAYS_PER_YEAR)) if sd > 1e-18 else np.nan


def annualized_return(r):
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n == 0:
        return np.nan
    eq = float(np.prod(1.0 + r))
    return float(eq ** (TRADING_DAYS_PER_YEAR / n) - 1.0) if eq > 0 else np.nan


def newey_west_mean_test(x, lag=2):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 5:
        return dict(T=T, mean=np.nan, se_hac=np.nan, t_hac=np.nan,
                    p_one_sided_gt0=np.nan, p_two_sided=np.nan)
    mu = float(np.mean(x))
    u = x - mu
    gamma0 = float(np.dot(u, u) / T)
    lrv = gamma0
    L = min(int(lag), T - 1)
    for ell in range(1, L + 1):
        gamma = float(np.dot(u[ell:], u[:-ell]) / T)
        weight = 1.0 - ell / (L + 1.0)
        lrv += 2.0 * weight * gamma
    lrv = max(lrv, 0.0)
    se = math.sqrt(lrv / T)
    if se <= 1e-18:
        t = p1 = p2 = np.nan
    else:
        t = mu / se
        p1 = float(1.0 - norm.cdf(t))
        p2 = float(2.0 * (1.0 - norm.cdf(abs(t))))
    return dict(T=T, mean=mu, se_hac=se, t_hac=t,
                p_one_sided_gt0=p1, p_two_sided=p2)

# =========================================================
# 3) PREDICTIONS + PRE-2025 CALIBRATION
# =========================================================

def load_pred(fp, model):
    d = safe_read_csv(fp)
    need = {"date", "stock", "y_pred", "y_true"}
    miss = need - set(d.columns)
    if miss:
        raise ValueError(f"{model} missing columns: {miss}")
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    d["stock"] = normalize_stock_series(d["stock"])
    d["y_pred"] = pd.to_numeric(d["y_pred"], errors="coerce")
    d["y_true"] = pd.to_numeric(d["y_true"], errors="coerce")
    d = d.dropna(subset=["date", "stock", "y_pred", "y_true"]).copy()
    d = d.drop_duplicates(["date", "stock"], keep="last")
    d["model"] = model
    return d


def chronological_calibration_slice(train_df):
    dates = np.array(sorted(train_df["date"].unique()))
    n = len(dates)
    n_cal = max(MIN_CALIBRATION_DATES, int(round(n * CALIBRATION_RATIO)))
    n_cal = min(n_cal, n)
    cal_dates = set(dates[-n_cal:])
    return train_df[train_df["date"].isin(cal_dates)].copy(), dates[-n_cal], dates[-1]


def fit_isotonic_calibrator(train_df, model_name, hurdle_decimal):
    cal, cal_start, cal_end = chronological_calibration_slice(train_df)

    x = cal["y_pred"].to_numpy(dtype=float)
    y = cal["y_true"].to_numpy(dtype=float)
    if Y_TRUE_IS_PERCENT:
        y = y / 100.0

    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 100:
        raise RuntimeError(f"Too few calibration observations for {model_name}: {len(x)}")

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(x, y)

    # Freeze a raw-score threshold from the PRE-2025 calibration map only.
    # Dense quantile grid is deterministic and does not inspect 2025 outcomes.
    q_grid = np.linspace(0.0, 1.0, 5001)
    score_grid = np.quantile(x, q_grid)
    pred_grid = iso.predict(score_grid)
    ok = pred_grid >= hurdle_decimal

    if np.any(ok):
        first = int(np.argmax(ok))
        raw_threshold = float(score_grid[first])
        calibrated_at_threshold = float(pred_grid[first])
        threshold_quantile = float(q_grid[first])
    else:
        # If no calibration point clears the fixed hurdle, the rule correctly
        # produces no trades rather than weakening the hurdle using 2025 data.
        raw_threshold = np.inf
        calibrated_at_threshold = np.nan
        threshold_quantile = 1.0

    # Tail diagnostics on PRE-2025 calibration data only.
    cal_expected = iso.predict(x)
    active = cal_expected >= hurdle_decimal
    tail = pd.DataFrame({
        "model": model_name,
        "n_cal_obs": [len(x)],
        "active_obs": [int(active.sum())],
        "active_share": [float(active.mean())],
        "mean_realized_gross_active": [float(y[active].mean()) if active.any() else np.nan],
        "median_realized_gross_active": [float(np.median(y[active])) if active.any() else np.nan],
        "mean_calibrated_expected_active": [float(cal_expected[active].mean()) if active.any() else np.nan],
    })

    summary = dict(
        model=model_name,
        calibration_start=str(pd.Timestamp(cal_start).date()),
        calibration_end=str(pd.Timestamp(cal_end).date()),
        calibration_dates=int(cal["date"].nunique()),
        calibration_obs=int(len(x)),
        hurdle_slippage_bps_per_side=int(CALIBRATION_SLIPPAGE_BPS_PER_SIDE),
        hurdle_gross_decimal=float(hurdle_decimal),
        hurdle_gross_percent=float(hurdle_decimal * 100.0),
        raw_y_pred_threshold=raw_threshold,
        threshold_quantile=threshold_quantile,
        calibrated_expected_at_threshold=calibrated_at_threshold,
    )
    return iso, summary, tail

# =========================================================
# 4) MARKET CACHE: SPLIT 14:55 INFORMATION FROM 15:00 FILL
# =========================================================

def load_market_cache():
    if not os.path.exists(MARKET_CACHE_CSV):
        raise FileNotFoundError(
            "market_execution_cache.csv not found. Run the previous "
            "Hybrid1455_ClosingAuction_Backtest_FINAL.py once to build it:\n"
            + MARKET_CACHE_CSV
        )
    m = safe_read_csv(MARKET_CACHE_CSV)
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.normalize()
    m["buy_date"] = pd.to_datetime(m["buy_date"], errors="coerce").dt.normalize()
    m["stock"] = normalize_stock_series(m["stock"])

    numeric_cols = [
        "prev_close", "price_1455", "entry_close", "exit_open",
        "ret_1455_vs_prev_close", "ret_1500_vs_prev_close",
        "gross_return_raw", "vol_to_1455", "auction_vol",
        "full_day_vol", "amount_to_1455", "auction_amount", "full_day_amount",
    ]
    for c in numeric_cols:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")

    # Whether this stock-day has volume data at all.
    vol_data = m.get("volume_data_available", pd.Series(False, index=m.index))
    if vol_data.dtype != bool:
        vol_data = vol_data.astype(str).str.lower().isin(["true", "1", "yes"])

    # PRE-TRADE eligibility: 14:55 information ONLY.
    price1455_ok = (
        m["price_1455"].notna() & (m["price_1455"] > 0)
        & m["prev_close"].notna() & (m["prev_close"] > 0)
    )
    limit1455_ok = (
        m["ret_1455_vs_prev_close"].notna()
        & (m["ret_1455_vs_prev_close"] < BUY_LIMIT_THRESHOLD)
    )
    volume_pre_ok = (~vol_data) | (m["vol_to_1455"].notna() & (m["vol_to_1455"] > 0))
    m["pretrade_eligible_1455"] = price1455_ok & limit1455_ok & volume_pre_ok

    # POST-SUBMISSION fill feasibility: 15:00 information. Never used to
    # rank or replace candidates.
    entry_ok = m["entry_close"].notna() & (m["entry_close"] > 0)
    limit1500_ok = (
        m["ret_1500_vs_prev_close"].notna()
        & (m["ret_1500_vs_prev_close"] < BUY_LIMIT_THRESHOLD)
    )
    auction_vol_ok = (~vol_data) | (m["auction_vol"].notna() & (m["auction_vol"] > 0))
    m["fill_ok_1500"] = entry_ok & limit1500_ok & auction_vol_ok

    # Exit is future realization, not a selection variable.
    m["exit_ok_next_open"] = m["exit_open"].notna() & (m["exit_open"] > 0)
    return m

# =========================================================
# 5) TEST MERGE + COST-AWARE MAX-K SELECTION
# =========================================================

def build_common_test(real_test, zero_test, market, real_iso, zero_iso, hurdle_decimal):
    r = real_test[["date", "stock", "y_pred", "y_true"]].rename(
        columns={"y_pred": "y_pred_real", "y_true": "y_true_real"})
    z = zero_test[["date", "stock", "y_pred", "y_true"]].rename(
        columns={"y_pred": "y_pred_zero", "y_true": "y_true_zero"})
    d = r.merge(z, on=["date", "stock"], how="inner")
    diff = (d["y_true_real"] - d["y_true_zero"]).abs()
    if len(diff) and float(diff.max()) > 1e-8:
        raise RuntimeError("Graph-2 and ZeroGraph y_true differ on common test sample")
    d["y_true"] = d["y_true_real"]
    d = d.drop(columns=["y_true_real", "y_true_zero"])
    d = d.merge(market, on=["date", "stock"], how="left")
    d["pretrade_eligible_1455"] = d["pretrade_eligible_1455"].fillna(False)
    d["fill_ok_1500"] = d["fill_ok_1500"].fillna(False)
    d["exit_ok_next_open"] = d["exit_ok_next_open"].fillna(False)

    # Calibrated expectations are functions of y_pred only, frozen pre-2025.
    d["exp_gross_real"] = real_iso.predict(d["y_pred_real"].to_numpy(dtype=float))
    d["exp_gross_zero"] = zero_iso.predict(d["y_pred_zero"].to_numpy(dtype=float))
    d["active_real"] = d["exp_gross_real"] >= hurdle_decimal
    d["active_zero"] = d["exp_gross_zero"] >= hurdle_decimal
    return d


def select_model_maxk(merged, model_name, pred_col, exp_col, active_col, k):
    signal_rows = []
    day_rows = []
    all_dates = sorted(merged["date"].dropna().unique())

    for date in all_dates:
        g = merged[
            (merged["date"] == date)
            & merged["pretrade_eligible_1455"].fillna(False)
            & merged[active_col].fillna(False)
        ].copy()
        g = g.dropna(subset=[pred_col, exp_col])
        g = g.sort_values([pred_col, "stock"], ascending=[False, True])
        sel = g.head(k).copy()

        submitted = int(len(sel))
        filled = 0
        sell_risk = 0
        missing_exit_after_fill = 0

        for rank, (_, r) in enumerate(sel.iterrows(), start=1):
            fill = bool(r["fill_ok_1500"])
            exit_ok = bool(r["exit_ok_next_open"])

            if fill and not exit_ok:
                missing_exit_after_fill += 1

            executed = bool(fill and exit_ok)
            if executed:
                filled += 1
                gross = float(r["gross_return_raw"])
                risk = bool(np.isfinite(gross) and gross <= SELL_LIMIT_RISK_THRESHOLD)
                sell_risk += int(risk)
            else:
                gross = np.nan
                risk = False

            signal_rows.append({
                "date": pd.Timestamp(date),
                "buy_date": r.get("buy_date", pd.NaT),
                "model": model_name,
                "K_max": int(k),
                "rank": int(rank),
                "stock": r["stock"],
                "y_pred": float(r[pred_col]),
                "calibrated_expected_gross": float(r[exp_col]),
                "submitted_at_1455": True,
                "fill_ok_1500": fill,
                "exit_ok_next_open": exit_ok,
                "executed": executed,
                "gross_return_raw": gross,
                "slot_weight": 1.0 / float(k),
                "sell_limit_risk": risk,
                "price_1455": r.get("price_1455", np.nan),
                "entry_close": r.get("entry_close", np.nan),
                "exit_open": r.get("exit_open", np.nan),
                "auction_vol": r.get("auction_vol", np.nan),
                "auction_amount": r.get("auction_amount", np.nan),
            })

        if missing_exit_after_fill > 0:
            raise RuntimeError(
                f"{model_name} K={k} {pd.Timestamp(date).date()}: "
                f"{missing_exit_after_fill} filled order(s) have no next-open exit. "
                "Do not silently drop them; inspect execution data."
            )

        day_rows.append({
            "date": pd.Timestamp(date),
            "model": model_name,
            "K_max": int(k),
            "submitted_signals": submitted,
            "filled_positions": filled,
            "cash_slots": int(k - filled),
            "cash_weight": float((k - filled) / k),
            "sell_limit_risk_count": sell_risk,
        })

    return pd.DataFrame(day_rows), pd.DataFrame(signal_rows)


def add_realized_returns(day_base, signals):
    rows = []
    sig_groups = {}
    if not signals.empty:
        for key, g in signals.groupby(["date", "model", "K_max"], sort=False):
            sig_groups[key] = g.copy()

    for _, d in day_base.iterrows():
        key = (pd.Timestamp(d["date"]), d["model"], int(d["K_max"]))
        g = sig_groups.get(key, pd.DataFrame())

        for bps in SLIPPAGE_BPS_LIST:
            if g.empty:
                port = 0.0
            else:
                exec_g = g[g["executed"] == True]
                if exec_g.empty:
                    port = 0.0
                else:
                    indiv_net = exact_net_return(
                        exec_g["gross_return_raw"].to_numpy(dtype=float), bps
                    )
                    # Each submitted portfolio has K equal-capital slots; unused
                    # or unfilled slots remain cash with zero return.
                    port = float(np.sum(indiv_net) / int(d["K_max"]))

            rows.append({
                **d.to_dict(),
                "slippage_bps_per_side": int(bps),
                "net_portfolio_return": port,
            })
    return pd.DataFrame(rows)

# =========================================================
# 6) SUMMARIES / HAC / DIAGNOSTIC GOAL CHECK
# =========================================================

def performance_summary(daily):
    rows = []
    for (model, k, bps), g in daily.groupby(
        ["model", "K_max", "slippage_bps_per_side"], sort=True
    ):
        g = g.sort_values("date")
        r = g["net_portfolio_return"].to_numpy(dtype=float)
        equity = np.cumprod(1.0 + r)
        rows.append({
            "model": model,
            "K": int(k),
            "slippage_bps_per_side": int(bps),
            "N_Days": int(len(g)),
            "DaysWithSubmittedSignal": int((g["submitted_signals"] > 0).sum()),
            "DaysWithFill": int((g["filled_positions"] > 0).sum()),
            "AvgSubmittedSignals": float(g["submitted_signals"].mean()),
            "AvgFilledPositions": float(g["filled_positions"].mean()),
            "AvgCashWeight": float(g["cash_weight"].mean()),
            "FinalEquity": float(equity[-1]) if len(equity) else np.nan,
            "TotalReturn": float(equity[-1] - 1.0) if len(equity) else np.nan,
            "AnnReturn": annualized_return(r),
            "Sharpe": sharpe_zero_rf(r),
            "MaxDrawdown": max_drawdown(equity),
            "WinRate_AllDays": float(np.mean(r > 0)) if len(r) else np.nan,
            "WinRate_ActiveDays": float(np.mean(r[g["filled_positions"].to_numpy() > 0] > 0))
                if (g["filled_positions"] > 0).any() else np.nan,
            "AvgDailyNetRet": float(np.mean(r)) if len(r) else np.nan,
            "SellLimitRiskEvents": int(g["sell_limit_risk_count"].sum()),
        })
    return pd.DataFrame(rows)


def hac_real_vs_zero(daily):
    rows = []
    for k in TOPK_LIST:
        for bps in SLIPPAGE_BPS_LIST:
            r = daily[(daily["model"] == "Graph-2") & (daily["K_max"] == k)
                      & (daily["slippage_bps_per_side"] == bps)][["date", "net_portfolio_return"]]
            z = daily[(daily["model"] == "ZeroGraph") & (daily["K_max"] == k)
                      & (daily["slippage_bps_per_side"] == bps)][["date", "net_portfolio_return"]]
            m = r.merge(z, on="date", suffixes=("_real", "_zero"), how="inner")
            diff = m["net_portfolio_return_real"] - m["net_portfolio_return_zero"]
            stat = newey_west_mean_test(diff.to_numpy(dtype=float), lag=HAC_LAG)
            rows.append({
                "K": k,
                "slippage_bps_per_side": bps,
                "HAC_lag": HAC_LAG,
                "mean_daily_real_minus_zero": stat["mean"],
                "se_hac": stat["se_hac"],
                "t_hac": stat["t_hac"],
                "p_one_sided_real_gt_zero": stat["p_one_sided_gt0"],
                "p_two_sided": stat["p_two_sided"],
                "T": stat["T"],
            })
    return pd.DataFrame(rows)


def goal_diagnostic(summary):
    s = summary[(summary["model"] == "Graph-2") & (summary["slippage_bps_per_side"] == 0)].copy()
    rows = []
    for _, r in s.iterrows():
        k = int(r["K"])
        tr = float(r["TotalReturn"])
        rows.append({
            "K": k,
            "Graph2_TotalReturn_0bp": tr,
            "PositiveReturn": bool(tr > 0),
            "OldPaper_K10_Reference": OLD_PAPER_K10_TOTAL_RETURN if k == 10 else np.nan,
            "K10_Above_OldPaper_48_38pct": bool(tr >= OLD_PAPER_K10_TOTAL_RETURN) if k == 10 else np.nan,
            "IMPORTANT": "DIAGNOSTIC ONLY - never feeds back into calibration or rule selection",
        })
    return pd.DataFrame(rows)

# =========================================================
# 7) MAIN
# =========================================================

def main():
    print("=== Hybrid1455 AE-CLEAN Cost-Aware Backtest ===")
    print("[NO 2025 TUNING] calibration uses PRE-2025 last 20% dates only")
    print("[TIMING] 14:55 rank/participate -> closing auction -> next open")
    print("[K] interpreted as MAXIMUM positions; unused slots remain cash")
    print("[NO EX-POST REPLACEMENT] 15:00 failed fill => cash slot")

    real_train_fp = resolve_existing(REAL_TRAIN_CANDIDATES, "Graph-2 train predictions")
    real_test_fp = resolve_existing(REAL_TEST_CANDIDATES, "Graph-2 test predictions")
    zero_train_fp = resolve_existing(ZERO_TRAIN_CANDIDATES, "ZeroGraph train predictions")
    zero_test_fp = resolve_existing(ZERO_TEST_CANDIDATES, "ZeroGraph test predictions")

    real_train = load_pred(real_train_fp, "Graph-2")
    real_test = load_pred(real_test_fp, "Graph-2")
    zero_train = load_pred(zero_train_fp, "ZeroGraph")
    zero_test = load_pred(zero_test_fp, "ZeroGraph")

    hurdle = exact_break_even_gross(CALIBRATION_SLIPPAGE_BPS_PER_SIDE)
    print(f"[FROZEN HURDLE] gross break-even @ base costs + {CALIBRATION_SLIPPAGE_BPS_PER_SIDE}bp/side = "
          f"{hurdle:.6%}")

    real_iso, real_cal, real_tail = fit_isotonic_calibrator(real_train, "Graph-2", hurdle)
    zero_iso, zero_cal, zero_tail = fit_isotonic_calibrator(zero_train, "ZeroGraph", hurdle)

    calibration_summary = pd.DataFrame([real_cal, zero_cal])
    calibration_tail = pd.concat([real_tail, zero_tail], ignore_index=True)
    calibration_summary.to_csv(os.path.join(OUT_DIR, "calibration_summary.csv"), index=False, encoding="utf-8-sig")
    calibration_tail.to_csv(os.path.join(OUT_DIR, "calibration_tail_diagnostics.csv"), index=False, encoding="utf-8-sig")

    print("\n========== FROZEN PRE-2025 CALIBRATION ==========")
    print(calibration_summary.to_string(index=False))
    print("\n========== PRE-2025 ACTIVE-TAIL DIAGNOSTIC ==========")
    print(calibration_tail.to_string(index=False))

    market = load_market_cache()
    merged = build_common_test(real_test, zero_test, market, real_iso, zero_iso, hurdle)

    day_parts = []
    sig_parts = []
    for model, pred_col, exp_col, active_col in [
        ("Graph-2", "y_pred_real", "exp_gross_real", "active_real"),
        ("ZeroGraph", "y_pred_zero", "exp_gross_zero", "active_zero"),
    ]:
        for k in TOPK_LIST:
            db, sg = select_model_maxk(merged, model, pred_col, exp_col, active_col, k)
            day_parts.append(db)
            sig_parts.append(sg)

    day_base = pd.concat(day_parts, ignore_index=True)
    signals = pd.concat(sig_parts, ignore_index=True) if sig_parts else pd.DataFrame()
    daily = add_realized_returns(day_base, signals)
    summary = performance_summary(daily)
    hac = hac_real_vs_zero(daily)
    goal = goal_diagnostic(summary)

    summary.to_csv(os.path.join(OUT_DIR, "summary_costaware.csv"), index=False, encoding="utf-8-sig")
    daily.to_csv(os.path.join(OUT_DIR, "daily_returns_costaware.csv"), index=False, encoding="utf-8-sig")
    signals.to_csv(os.path.join(OUT_DIR, "selected_signals_costaware.csv"), index=False, encoding="utf-8-sig")
    hac.to_csv(os.path.join(OUT_DIR, "real_zero_daily_diff_hac_costaware.csv"), index=False, encoding="utf-8-sig")
    goal.to_csv(os.path.join(OUT_DIR, "goal_diagnostic_only.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    print("\n========== PERFORMANCE SUMMARY ==========")
    cols = [
        "model", "K", "slippage_bps_per_side", "DaysWithSubmittedSignal",
        "DaysWithFill", "AvgFilledPositions", "AvgCashWeight", "FinalEquity",
        "TotalReturn", "Sharpe", "MaxDrawdown", "WinRate_ActiveDays",
        "SellLimitRiskEvents",
    ]
    print(summary[cols].to_string(index=False))

    print("\n========== GRAPH-2 - ZEROGRAPH HAC ==========")
    print(hac.to_string(index=False))

    print("\n========== GOAL DIAGNOSTIC ONLY ==========")
    print(goal.to_string(index=False))
    print("\nIMPORTANT: goal_diagnostic_only.csv is evaluation only. Do not alter the")
    print("pre-2025 calibration rule after seeing 2025 results.")

    print("\nSaved to:")
    print(OUT_DIR)
    print("\nPlease send back these 5 files:")
    print("1) calibration_summary.csv")
    print("2) calibration_tail_diagnostics.csv")
    print("3) summary_costaware.csv")
    print("4) real_zero_daily_diff_hac_costaware.csv")
    print("5) goal_diagnostic_only.csv")


if __name__ == "__main__":
    main()
