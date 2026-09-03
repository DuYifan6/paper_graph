# -*- coding: utf-8 -*-


import os
import math
import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm

# ============================================================
# 0) PATHS
# ============================================================

BASE = (
    r"D:\PythonProject\LASSO_FINAL\graph_backtest"
    r"\economic_executable\hybrid1455"
)

SOURCE_DIR = os.path.join(
    BASE, "closing_auction_costaware_ae_clean"
)

SELECTED_CSV = os.path.join(
    SOURCE_DIR, "selected_signals_costaware.csv"
)

DAILY_SOURCE_CSV = os.path.join(
    SOURCE_DIR, "daily_returns_costaware.csv"
)

COMMON_CACHE_NPZ = (
    r"D:\PythonProject\LASSO_FINAL\common_preprocess_cache"
    r"\common_tensor_cache.npz"
)

META_PKL = (
    r"D:\PythonProject\LASSO_FINAL\common_preprocess_cache"
    r"\common_tensor_meta.pkl"
)

OUT_DIR = os.path.join(
    BASE, "closing_auction_costaware_inverse_rank_volscaled"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1) FROZEN SETTINGS
# ============================================================

TOPK_LIST = [1, 3, 5, 10]
SLIPPAGE_BPS_LIST = [0, 5, 10]

BUY_COST = 0.0001
SELL_COST = 0.0011

TRADING_DAYS_PER_YEAR = 252
HAC_LAG = 2

# Same 20-day horizon as the forecasting setup.
RISK_LOOKBACK = 20

# Fixed PRE-2025 reference statistic.
# Do NOT change this after inspecting 2025 performance.
RISK_TARGET_STAT = "mean"

# No leverage.
MAX_EXPOSURE = 1.0

# y_np is stored in percentage points in the common cache.
Y_IS_PERCENT = True


# ============================================================
# 2) HELPERS
# ============================================================

def safe_read_csv(fp):
    last = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312"]:
        try:
            return pd.read_csv(fp, encoding=enc, low_memory=False)
        except Exception as e:
            last = e
    raise last


def exact_net_return(gross_ret, slip_bps):
    """
    Per-stock exact overnight net return.

    Entry cost:
        Close * (1 + BUY_COST) * (1 + slip)

    Exit proceeds:
        Open * (1 - SELL_COST - slip)
    """
    slip = float(slip_bps) / 10000.0
    buy_den = (1.0 + BUY_COST) * (1.0 + slip)
    sell_mult = 1.0 - SELL_COST - slip

    return (
        (1.0 + np.asarray(gross_ret, dtype=float))
        * sell_mult
        / buy_den
        - 1.0
    )


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
    if sd <= 1e-18:
        return np.nan
    return float(np.mean(r) / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


def annualized_return(r):
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return np.nan
    final = float(np.prod(1.0 + r))
    if final <= 0:
        return np.nan
    return float(final ** (TRADING_DAYS_PER_YEAR / len(r)) - 1.0)


def newey_west_mean_test(x, lag=2):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    T = len(x)
    if T < 5:
        return {
            "T": T,
            "mean": np.nan,
            "se_hac": np.nan,
            "t_hac": np.nan,
            "p_one_sided_gt0": np.nan,
            "p_two_sided": np.nan,
        }

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
        t = np.nan
        p1 = np.nan
        p2 = np.nan
    else:
        t = mu / se
        p1 = float(1.0 - norm.cdf(t))
        p2 = float(2.0 * (1.0 - norm.cdf(abs(t))))

    return {
        "T": T,
        "mean": mu,
        "se_hac": se,
        "t_hac": t,
        "p_one_sided_gt0": p1,
        "p_two_sided": p2,
    }


# ============================================================
# 3) BUILD STRICTLY LAGGED MARKET-RISK SCALER
# ============================================================

def build_risk_scaler():
    if not os.path.exists(COMMON_CACHE_NPZ):
        raise FileNotFoundError(COMMON_CACHE_NPZ)
    if not os.path.exists(META_PKL):
        raise FileNotFoundError(META_PKL)

    pack = np.load(COMMON_CACHE_NPZ)

    if "y_np" not in pack.files:
        raise KeyError("common_tensor_cache.npz has no y_np")

    y = pack["y_np"].astype(np.float64)

    if "mask_np" in pack.files:
        mask = pack["mask_np"].astype(bool)
    else:
        mask = np.isfinite(y)

    with open(META_PKL, "rb") as f:
        meta = pickle.load(f)

    dates = pd.to_datetime(meta["all_dates"]).normalize()

    if len(dates) != y.shape[0]:
        raise RuntimeError(
            f"date/y mismatch: {len(dates)} vs {y.shape[0]}"
        )

    # Market-wide overnight-gap volatility used in the robustness analysis:
    # V_t = sqrt(mean_i(y_it^2)).
    # Scale does not matter for exposure ratio, but convert to decimal
    # for more intuitive diagnostics.
    if Y_IS_PERCENT:
        y_dec = y / 100.0
    else:
        y_dec = y

    v = np.full(len(dates), np.nan, dtype=float)

    for t in range(len(dates)):
        ok = mask[t] & np.isfinite(y_dec[t])
        if ok.sum() < 50:
            continue
        v[t] = float(np.sqrt(np.mean(y_dec[t, ok] ** 2)))

    risk = pd.DataFrame({
        "date": dates,
        "market_gap_vol": v,
    }).sort_values("date").reset_index(drop=True)

    # For a target date t, use ONLY realized market-gap volatility through t-1.
    risk["trailing_market_gap_vol"] = (
        risk["market_gap_vol"]
        .shift(1)
        .rolling(RISK_LOOKBACK, min_periods=RISK_LOOKBACK)
        .mean()
    )

    pre = risk[
        (risk["date"] < pd.Timestamp("2025-01-01"))
        & risk["trailing_market_gap_vol"].notna()
    ].copy()

    if pre.empty:
        raise RuntimeError("No valid pre-2025 risk observations.")

    if RISK_TARGET_STAT == "mean":
        target = float(pre["trailing_market_gap_vol"].mean())
    elif RISK_TARGET_STAT == "median":
        target = float(pre["trailing_market_gap_vol"].median())
    else:
        raise ValueError("RISK_TARGET_STAT must be mean or median")

    risk["pre2025_target_risk"] = target

    risk["vol_exposure"] = np.where(
        risk["trailing_market_gap_vol"].notna()
        & (risk["trailing_market_gap_vol"] > 0),
        np.minimum(
            MAX_EXPOSURE,
            target / risk["trailing_market_gap_vol"],
        ),
        MAX_EXPOSURE,
    )

    risk["vol_exposure"] = np.clip(
        risk["vol_exposure"], 0.0, MAX_EXPOSURE
    )

    risk["is_2025"] = risk["date"].dt.year.eq(2025)

    print("\n========== RISK SCALER ==========")
    print("lookback =", RISK_LOOKBACK)
    print("target statistic =", RISK_TARGET_STAT)
    print("pre-2025 target risk =", target)

    r25 = risk[risk["is_2025"]]
    if not r25.empty:
        print("2025 mean exposure =", float(r25["vol_exposure"].mean()))
        print("2025 min exposure  =", float(r25["vol_exposure"].min()))
        print(
            "2025 days exposure < 1 =",
            int((r25["vol_exposure"] < 0.999999).sum()),
        )

    return risk


# ============================================================
# 4) INVERSE-RANK + VOL-SCALED DAILY RETURNS
# ============================================================

def build_daily_returns(signals, source_daily, risk):
    signals = signals.copy()
    signals["date"] = pd.to_datetime(
        signals["date"], errors="coerce"
    ).dt.normalize()

    # Robust bool parsing.
    for c in [
        "submitted_at_1455",
        "fill_ok_1500",
        "exit_ok_next_open",
        "executed",
    ]:
        if c in signals.columns and signals[c].dtype != bool:
            signals[c] = (
                signals[c]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            )

    source_daily = source_daily.copy()
    source_daily["date"] = pd.to_datetime(
        source_daily["date"], errors="coerce"
    ).dt.normalize()

    all_dates = sorted(
        source_daily["date"].dropna().unique()
    )

    risk_map = (
        risk.set_index("date")["vol_exposure"].to_dict()
    )

    rows = []

    models = sorted(signals["model"].dropna().unique())

    for model in models:
        for k in TOPK_LIST:
            sk = signals[
                (signals["model"] == model)
                & (signals["K_max"] == k)
            ].copy()

            for date in all_dates:
                d = pd.Timestamp(date)

                g = sk[sk["date"] == d].copy()
                sub = g[
                    g["submitted_at_1455"].fillna(False)
                ].copy()

                exposure = float(
                    risk_map.get(d, MAX_EXPOSURE)
                )

                # The inverse-rank weights are fixed at 14:55 among
                # SUBMITTED signals, before auction fill is observed.
                if len(sub) > 0:
                    rank = pd.to_numeric(
                        sub["rank"], errors="coerce"
                    ).astype(float)

                    inv = 1.0 / rank
                    denom = float(inv.sum())

                    if (
                        not np.isfinite(denom)
                        or denom <= 0
                    ):
                        raise RuntimeError(
                            f"Invalid rank denominator: "
                            f"{model}, K={k}, date={d}"
                        )

                    sub["base_rank_weight"] = inv / denom
                    sub["scaled_order_weight"] = (
                        exposure * sub["base_rank_weight"]
                    )
                else:
                    sub["base_rank_weight"] = []
                    sub["scaled_order_weight"] = []

                for bps in SLIPPAGE_BPS_LIST:
                    if len(sub) == 0:
                        port_ret_unscaled = 0.0
                        port_ret_scaled = 0.0
                        executed_count = 0
                        executed_weight_unscaled = 0.0
                        executed_weight_scaled = 0.0
                    else:
                        ex = sub[
                            sub["executed"].fillna(False)
                            & sub["gross_return_raw"].notna()
                        ].copy()

                        executed_count = int(len(ex))

                        if len(ex) == 0:
                            port_ret_unscaled = 0.0
                            port_ret_scaled = 0.0
                            executed_weight_unscaled = 0.0
                            executed_weight_scaled = 0.0
                        else:
                            indiv = exact_net_return(
                                ex["gross_return_raw"]
                                .to_numpy(dtype=float),
                                bps,
                            )

                            # AE-clean rule:
                            # DO NOT renormalize after 15:00 fill.
                            w0 = ex[
                                "base_rank_weight"
                            ].to_numpy(dtype=float)

                            ws = ex[
                                "scaled_order_weight"
                            ].to_numpy(dtype=float)

                            port_ret_unscaled = float(
                                np.sum(w0 * indiv)
                            )

                            port_ret_scaled = float(
                                np.sum(ws * indiv)
                            )

                            executed_weight_unscaled = float(
                                np.sum(w0)
                            )
                            executed_weight_scaled = float(
                                np.sum(ws)
                            )

                    rows.append({
                        "date": d,
                        "model": model,
                        "K": int(k),
                        "slippage_bps_per_side": int(bps),
                        "submitted_signals": int(len(sub)),
                        "executed_positions": executed_count,

                        "vol_exposure": exposure,

                        "executed_weight_unscaled": (
                            executed_weight_unscaled
                        ),
                        "executed_weight_volscaled": (
                            executed_weight_scaled
                        ),

                        "cash_weight_unscaled": float(
                            1.0 - executed_weight_unscaled
                        ),
                        "cash_weight_volscaled": float(
                            1.0 - executed_weight_scaled
                        ),

                        "net_ret_inverse_rank_unscaled": (
                            port_ret_unscaled
                        ),
                        "net_ret_inverse_rank_volscaled": (
                            port_ret_scaled
                        ),
                    })

    return pd.DataFrame(rows)


# ============================================================
# 5) PERFORMANCE SUMMARY
# ============================================================

def summarize(daily):
    rows = []

    for (
        model,
        k,
        bps,
    ), g in daily.groupby(
        ["model", "K", "slippage_bps_per_side"],
        sort=True,
    ):
        g = g.sort_values("date")

        for mode, col in [
            (
                "inverse_rank_unscaled",
                "net_ret_inverse_rank_unscaled",
            ),
            (
                "inverse_rank_volscaled",
                "net_ret_inverse_rank_volscaled",
            ),
        ]:
            r = g[col].to_numpy(dtype=float)
            eq = np.cumprod(1.0 + r)

            rows.append({
                "allocation": mode,
                "model": model,
                "K": int(k),
                "slippage_bps_per_side": int(bps),
                "N_Days": int(len(g)),
                "FinalEquity": (
                    float(eq[-1])
                    if len(eq) else np.nan
                ),
                "TotalReturn": (
                    float(eq[-1] - 1.0)
                    if len(eq) else np.nan
                ),
                "AnnReturn": annualized_return(r),
                "Sharpe": sharpe_zero_rf(r),
                "MaxDrawdown": max_drawdown(eq),
                "WinRate_AllDays": float(
                    np.mean(r > 0)
                ) if len(r) else np.nan,
                "AvgDailyNetRet": float(
                    np.mean(r)
                ) if len(r) else np.nan,
                "AvgVolExposure": float(
                    g["vol_exposure"].mean()
                ),
                "MinVolExposure": float(
                    g["vol_exposure"].min()
                ),
                "AvgCashWeight": float(
                    g[
                        "cash_weight_volscaled"
                        if mode == "inverse_rank_volscaled"
                        else "cash_weight_unscaled"
                    ].mean()
                ),
            })

    return pd.DataFrame(rows)


# ============================================================
# 6) HAC: GRAPH-2 - ZEROGRAPH
# ============================================================

def hac_table(daily):
    rows = []

    for mode, col in [
        (
            "inverse_rank_unscaled",
            "net_ret_inverse_rank_unscaled",
        ),
        (
            "inverse_rank_volscaled",
            "net_ret_inverse_rank_volscaled",
        ),
    ]:
        for k in TOPK_LIST:
            for bps in SLIPPAGE_BPS_LIST:
                r = daily[
                    (daily["model"] == "Graph-2")
                    & (daily["K"] == k)
                    & (
                        daily["slippage_bps_per_side"]
                        == bps
                    )
                ][["date", col]].rename(
                    columns={col: "real_ret"}
                )

                z = daily[
                    (daily["model"] == "ZeroGraph")
                    & (daily["K"] == k)
                    & (
                        daily["slippage_bps_per_side"]
                        == bps
                    )
                ][["date", col]].rename(
                    columns={col: "zero_ret"}
                )

                m = r.merge(z, on="date", how="inner")
                m["diff"] = (
                    m["real_ret"] - m["zero_ret"]
                )

                stat = newey_west_mean_test(
                    m["diff"].to_numpy(),
                    lag=HAC_LAG,
                )

                rows.append({
                    "allocation": mode,
                    "K": int(k),
                    "slippage_bps_per_side": int(bps),
                    "HAC_lag": HAC_LAG,
                    "mean_daily_graph2_minus_zero": stat[
                        "mean"
                    ],
                    "se_hac": stat["se_hac"],
                    "t_hac": stat["t_hac"],
                    "p_one_sided_graph2_gt_zero": stat[
                        "p_one_sided_gt0"
                    ],
                    "p_two_sided": stat["p_two_sided"],
                    "T": stat["T"],
                })

    return pd.DataFrame(rows)


# ============================================================
# 7) MAIN
# ============================================================

def main():
    if not os.path.exists(SELECTED_CSV):
        raise FileNotFoundError(SELECTED_CSV)
    if not os.path.exists(DAILY_SOURCE_CSV):
        raise FileNotFoundError(DAILY_SOURCE_CSV)

    signals = safe_read_csv(SELECTED_CSV)
    source_daily = safe_read_csv(DAILY_SOURCE_CSV)

    need = {
        "date",
        "model",
        "K_max",
        "rank",
        "submitted_at_1455",
        "executed",
        "gross_return_raw",
    }
    miss = need - set(signals.columns)
    if miss:
        raise ValueError(
            f"selected_signals_costaware.csv missing: {miss}"
        )

    risk = build_risk_scaler()

    daily = build_daily_returns(
        signals=signals,
        source_daily=source_daily,
        risk=risk,
    )

    summary = summarize(daily)
    hac = hac_table(daily)

    comparison = summary.pivot_table(
        index=[
            "model",
            "K",
            "slippage_bps_per_side",
        ],
        columns="allocation",
        values=[
            "TotalReturn",
            "Sharpe",
            "MaxDrawdown",
        ],
        aggfunc="first",
    )
    comparison.columns = [
        f"{a}_{b}" for a, b in comparison.columns
    ]
    comparison = comparison.reset_index()

    risk.to_csv(
        os.path.join(
            OUT_DIR, "risk_scaler_by_date.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    daily.to_csv(
        os.path.join(
            OUT_DIR, "daily_inverse_rank_volscaled.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        os.path.join(
            OUT_DIR, "summary_inverse_rank_volscaled.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    hac.to_csv(
        os.path.join(
            OUT_DIR,
            "real_zero_hac_inverse_rank_volscaled.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    comparison.to_csv(
        os.path.join(
            OUT_DIR,
            "comparison_unscaled_vs_volscaled.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    print("\n========== GRAPH-2 | 0 bp ==========")
    show = summary[
        (summary["model"] == "Graph-2")
        & (summary["slippage_bps_per_side"] == 0)
    ][
        [
            "allocation",
            "K",
            "TotalReturn",
            "Sharpe",
            "MaxDrawdown",
            "AvgVolExposure",
            "AvgCashWeight",
        ]
    ]
    print(show.to_string(index=False))

    print("\n========== COMPARISON ==========")
    print(comparison.to_string(index=False))

    print("\nSaved to:")
    print(OUT_DIR)

    print("\nPlease send back:")
    print("1) summary_inverse_rank_volscaled.csv")
    print("2) comparison_unscaled_vs_volscaled.csv")
    print("3) real_zero_hac_inverse_rank_volscaled.csv")
    print("4) risk_scaler_by_date.csv")


if __name__ == "__main__":
    main()
