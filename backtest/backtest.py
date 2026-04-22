# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1) 路径
# =========================================================
# 改这里：切换 A / B / C 的 Real 结果
REAL_PRED_CSV = r"D:\PythonProject\LASSO_FINAL\graph_调整\graph1_stronggraph\results\predictions_Graph1_C_K3510_test_detail.csv"

# 改这里：Zero 的结果
ZERO_PRED_CSV = r"D:\PythonProject\LASSO_FINAL\graph_调整\fixed_zerograph_valselect\results\predictions_ZeroGraph_test_detail.csv"

# 14:30 可交易过滤表
CLOSE_RET_WIDE_CSV = r"D:\PythonProject\LASSO_FINAL\data\14.30相对前一日收盘涨幅.csv"

# 输出目录
OUT_DIR = r"D:\PythonProject\LASSO_FINAL\backtest\common_sample_real_vs_zero_graph1"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# 2) 参数
# =========================================================
TOPK_LIST = [1, 3, 5, 10, 15, 20]
TRADING_DAYS_PER_YEAR = 252

# y_true=1.25 表示 1.25%
Y_TRUE_IS_PERCENT = True

# 14:30 涨幅表里 0.091 表示 9.1%
CLOSE_RET_IS_DECIMAL = True

# t-1 日 14:30 涨幅过滤阈值
LIMIT_THRESHOLD = 0.09 if CLOSE_RET_IS_DECIMAL else 9.0

# 缺失 14:30 涨幅时是否剔除
DROP_IF_CLOSE_RET_MISSING = True

# 成本
BUY_COST = 0.0001
SELL_COST = 0.0011
ROUNDTRIP_COST = BUY_COST + SELL_COST


# =========================================================
# 3) 工具函数
# =========================================================
def safe_read_csv(fp):
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312"]:
        try:
            return pd.read_csv(fp, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise last_err


def parse_trade_date_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)

    dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    miss = dt.isna()
    if miss.any():
        dt2 = pd.to_datetime(s[miss], errors="coerce")
        dt.loc[miss] = dt2
    return dt.dt.normalize()


def normalize_stock_code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())


def annualized_return(daily_ret: np.ndarray, n_days: int, trading_days=252) -> float:
    if n_days <= 0:
        return 0.0
    equity = np.cumprod(1.0 + daily_ret)
    total = equity[-1] - 1.0
    return float((1.0 + total) ** (trading_days / n_days) - 1.0)


def annualized_vol(daily_ret: np.ndarray, trading_days=252) -> float:
    if daily_ret.size <= 1:
        return 0.0
    return float(np.std(daily_ret, ddof=1) * math.sqrt(trading_days))


def sharpe_ratio(daily_ret: np.ndarray, trading_days=252, rf=0.0) -> float:
    if daily_ret.size <= 1:
        return 0.0
    mu = np.mean(daily_ret) - rf / trading_days
    sd = np.std(daily_ret, ddof=1)
    return float(mu / (sd + 1e-12) * math.sqrt(trading_days))


def summarize_backtest(df_daily: pd.DataFrame) -> dict:
    r = df_daily["net_ret"].values.astype(float)
    n = len(r)
    eq = df_daily["equity"].values.astype(float)

    mdd = max_drawdown(eq)
    ann_ret = annualized_return(r, n, TRADING_DAYS_PER_YEAR)
    ann_vol = annualized_vol(r, TRADING_DAYS_PER_YEAR)
    sharpe = sharpe_ratio(r, TRADING_DAYS_PER_YEAR, rf=0.0)
    calmar = float(ann_ret / (abs(mdd) + 1e-12))

    trade_mask = df_daily["n_hold"].values > 0
    if trade_mask.sum() > 0:
        win_rate = float((df_daily.loc[trade_mask, "net_ret"] > 0).mean())
        avg_trade_ret = float(df_daily.loc[trade_mask, "net_ret"].mean())
        avg_trade_gross = float(df_daily.loc[trade_mask, "gross_ret"].mean())
    else:
        win_rate = 0.0
        avg_trade_ret = 0.0
        avg_trade_gross = 0.0

    return {
        "Days": int(n),
        "TradeDays": int(trade_mask.sum()),
        "NoTradeDays": int((~trade_mask).sum()),
        "FinalEquity": float(eq[-1]) if n > 0 else 1.0,
        "TotalReturn": float(eq[-1] - 1.0) if n > 0 else 0.0,
        "AnnReturn": ann_ret,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDrawdown": mdd,
        "Calmar": calmar,
        "WinRate": win_rate,
        "AvgDailyRet": float(np.mean(r)) if n > 0 else 0.0,
        "StdDailyRet": float(np.std(r, ddof=1)) if n > 1 else 0.0,
        "AvgTradeDayRet": avg_trade_ret,
        "AvgTradeDayGrossRet": avg_trade_gross,
    }


def format_summary_for_display(df_sum: pd.DataFrame) -> pd.DataFrame:
    out = df_sum.copy()

    pct_cols = [
        "TotalReturn", "AnnReturn", "AnnVol", "MaxDrawdown",
        "AvgDailyRet", "StdDailyRet", "AvgTradeDayRet", "AvgTradeDayGrossRet"
    ]
    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:.4%}")

    ratio_cols = ["Sharpe", "Calmar", "WinRate", "FinalEquity", "AvgHold"]
    for c in ratio_cols:
        if c in out.columns:
            if c == "WinRate":
                out[c] = out[c].map(lambda x: f"{x:.2%}")
            elif c == "FinalEquity":
                out[c] = out[c].map(lambda x: f"{x:.4f}")
            else:
                out[c] = out[c].map(lambda x: f"{x:.4f}")
    return out


# =========================================================
# 4) 读取预测表
# =========================================================
def load_prediction_table(pred_path: str, model_name: str):
    df = safe_read_csv(pred_path)

    need_cols = {"date", "stock", "y_pred", "y_true"}
    miss = need_cols - set(df.columns)
    if miss:
        raise ValueError(f"{model_name} 预测结果表缺少列: {miss}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["stock"] = normalize_stock_code(df["stock"])
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df = df.dropna(subset=["date", "stock", "y_pred", "y_true"]).copy()

    if Y_TRUE_IS_PERCENT:
        df["ret_gap"] = df["y_true"] / 100.0
    else:
        df["ret_gap"] = df["y_true"].astype(float)

    df["model"] = model_name
    return df


def build_common_prediction_table(df_real: pd.DataFrame, df_zero: pd.DataFrame) -> pd.DataFrame:
    real = df_real[["date", "stock", "y_pred", "y_true", "ret_gap"]].copy()
    real = real.rename(columns={
        "y_pred": "y_pred_real",
        "y_true": "y_true_real",
        "ret_gap": "ret_gap_real",
    })

    zero = df_zero[["date", "stock", "y_pred", "y_true", "ret_gap"]].copy()
    zero = zero.rename(columns={
        "y_pred": "y_pred_zero",
        "y_true": "y_true_zero",
        "ret_gap": "ret_gap_zero",
    })

    merged = real.merge(zero, on=["date", "stock"], how="inner")

    merged["y_true_diff"] = (merged["y_true_real"] - merged["y_true_zero"]).abs()
    bad = merged["y_true_diff"] > 1e-10
    if bad.any():
        print(f"[警告] 共同样本上发现 {bad.sum()} 行 y_true 不一致，将保留 real 的 y_true 作为统一真值。")

    merged["y_true"] = merged["y_true_real"]
    merged["ret_gap"] = merged["ret_gap_real"]
    return merged


# =========================================================
# 5) 读取 14:30 涨幅表
# =========================================================
def load_close_ret_wide(close_ret_path: str):
    wide = safe_read_csv(close_ret_path)
    wide.columns = [str(c).strip() for c in wide.columns]

    date_col = None
    for c in ["trade_date", "date", "Date", "TRADE_DATE"]:
        if c in wide.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"收盘涨幅表中未找到日期列，当前列名: {wide.columns.tolist()[:20]}")

    wide[date_col] = parse_trade_date_series(wide[date_col])
    wide = wide.dropna(subset=[date_col]).copy()

    stock_cols = [c for c in wide.columns if c != date_col]
    if len(stock_cols) == 0:
        raise ValueError("收盘涨幅表除日期列外，没有股票列")

    long_df = (
        wide.set_index(date_col)[stock_cols]
        .stack(dropna=True)
        .reset_index()
    )
    long_df.columns = ["buy_date", "stock", "close_ret"]

    long_df["stock"] = normalize_stock_code(long_df["stock"])
    long_df["close_ret"] = pd.to_numeric(long_df["close_ret"], errors="coerce")
    long_df = long_df.dropna(subset=["buy_date", "stock", "close_ret"]).copy()

    return long_df


# =========================================================
# 6) 给预测日匹配 buy_date=t-1
# =========================================================
def attach_buy_date(df_pred: pd.DataFrame, df_close_long: pd.DataFrame) -> pd.DataFrame:
    out = df_pred.copy()

    buy_dates = np.array(
        sorted(pd.to_datetime(df_close_long["buy_date"].dropna().unique())),
        dtype="datetime64[ns]"
    )
    pred_dates = out["date"].values.astype("datetime64[ns]")

    idx = np.searchsorted(buy_dates, pred_dates, side="left") - 1
    buy_date_arr = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")

    valid = idx >= 0
    buy_date_arr[valid] = buy_dates[idx[valid]]

    out["buy_date"] = pd.to_datetime(buy_date_arr)
    out["buy_date_missing"] = out["buy_date"].isna().astype(int)
    return out


# =========================================================
# 7) 先过滤共同可交易池
# =========================================================
def build_tradeable_common_universe(df_common: pd.DataFrame, df_close_long: pd.DataFrame):
    merged = df_common.merge(
        df_close_long,
        on=["buy_date", "stock"],
        how="left"
    )

    merged["close_ret_missing"] = merged["close_ret"].isna().astype(int)
    merged["filtered_by_limit"] = np.where(
        (merged["close_ret"].notna()) & (merged["close_ret"] >= LIMIT_THRESHOLD),
        1, 0
    )

    if DROP_IF_CLOSE_RET_MISSING:
        merged["is_tradeable"] = np.where(
            (merged["close_ret"].notna()) & (merged["close_ret"] < LIMIT_THRESHOLD),
            1, 0
        )
    else:
        merged["is_tradeable"] = np.where(
            (merged["close_ret"].isna()) | (merged["close_ret"] < LIMIT_THRESHOLD),
            1, 0
        )

    tradeable = merged[merged["is_tradeable"] == 1].copy()
    return merged, tradeable


# =========================================================
# 8) 在共同可交易池里分别回测
# =========================================================
def backtest_filter_first_topk_common(df_tradeable: pd.DataFrame, all_pred_dates: list, k: int, model: str):
    pred_col = "y_pred_real" if model.lower() == "realgraph" else "y_pred_zero"

    df = df_tradeable.sort_values(["date", pred_col, "stock"], ascending=[True, False, True]).copy()
    df["rank_after_filter"] = df.groupby("date")[pred_col].rank(method="first", ascending=False).astype(int)
    df_k = df[df["rank_after_filter"] <= k].copy()

    grouped = {d: sub.copy() for d, sub in df_k.groupby("date", sort=True)}

    daily_rows = []
    trade_rows = []

    for d in all_pred_dates:
        d = pd.Timestamp(d)
        sub = grouped.get(d, None)

        if sub is None or len(sub) == 0:
            daily_rows.append({
                "date": d,
                "buy_date": pd.NaT,
                "model": model,
                "K": k,
                "n_hold": 0,
                "gross_ret": 0.0,
                "net_ret": 0.0,
                "buy_cost": 0.0,
                "sell_cost": 0.0,
                "roundtrip_cost": 0.0,
                "is_trade_day": 0,
            })
            continue

        rets = sub["ret_gap"].values.astype(float)
        n_hold = len(rets)
        gross = float(np.mean(rets))
        net = gross - ROUNDTRIP_COST

        daily_rows.append({
            "date": d,
            "buy_date": sub["buy_date"].iloc[0] if "buy_date" in sub.columns else pd.NaT,
            "model": model,
            "K": k,
            "n_hold": int(n_hold),
            "gross_ret": gross,
            "net_ret": net,
            "buy_cost": BUY_COST,
            "sell_cost": SELL_COST,
            "roundtrip_cost": ROUNDTRIP_COST,
            "is_trade_day": 1,
        })

        for _, r in sub.iterrows():
            trade_rows.append({
                "date": pd.Timestamp(r["date"]),
                "buy_date": pd.Timestamp(r["buy_date"]) if pd.notna(r["buy_date"]) else pd.NaT,
                "model": model,
                "K": k,
                "stock": r["stock"],
                "rank_after_filter": int(r["rank_after_filter"]),
                "y_pred_model": float(r[pred_col]),
                "y_pred_real": float(r["y_pred_real"]),
                "y_pred_zero": float(r["y_pred_zero"]),
                "y_true": float(r["y_true"]),
                "close_ret": float(r["close_ret"]) if pd.notna(r["close_ret"]) else np.nan,
                "ret_gross": float(r["ret_gap"]),
                "buy_cost": BUY_COST,
                "sell_cost": SELL_COST,
                "ret_net": float(r["ret_gap"] - ROUNDTRIP_COST),
            })

    df_daily = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    df_daily["equity"] = (1.0 + df_daily["net_ret"]).cumprod()

    df_trades = pd.DataFrame(trade_rows)
    if not df_trades.empty:
        df_trades = df_trades.sort_values(["date", "rank_after_filter"]).reset_index(drop=True)

    return df_daily, df_trades


# =========================================================
# 9) 画图
# =========================================================
def plot_equity_compare(d_real: pd.DataFrame, d_zero: pd.DataFrame, out_png: str, title: str):
    plt.figure(figsize=(10, 5))
    plt.plot(d_real["date"], d_real["equity"], label=f"RealGraph")
    plt.plot(d_zero["date"], d_zero["equity"], label=f"ZeroGraph")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# =========================================================
# 10) 主程序
# =========================================================
def run_all():
    print("=== Step 1: 读取 Real / Zero 预测结果 ===")
    df_real = load_prediction_table(REAL_PRED_CSV, "RealGraph")
    df_zero = load_prediction_table(ZERO_PRED_CSV, "ZeroGraph")
    print(f"RealGraph 样本数: {len(df_real)}")
    print(f"ZeroGraph 样本数: {len(df_zero)}")

    print("\n=== Step 2: 构造 common sample ===")
    df_common = build_common_prediction_table(df_real, df_zero)
    print(f"共同样本数: {len(df_common)}")
    print(f"共同预测日数: {df_common['date'].nunique()}")
    print(f"共同股票数: {df_common['stock'].nunique()}")

    all_pred_dates = sorted(df_common["date"].dropna().unique().tolist())
    df_common.to_csv(
        os.path.join(OUT_DIR, "common_sample_predictions.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== Step 3: 读取 14:30 涨幅表 ===")
    df_close_long = load_close_ret_wide(CLOSE_RET_WIDE_CSV)
    print(f"14:30 涨幅长表记录数: {len(df_close_long)}")

    print("\n=== Step 4: 生成 buy_date=t-1 ===")
    df_common = attach_buy_date(df_common, df_close_long)
    print(f"buy_date 缺失样本数: {int(df_common['buy_date_missing'].sum())}")

    print("\n=== Step 5: 过滤共同可交易池 ===")
    full_universe, tradeable_universe = build_tradeable_common_universe(df_common, df_close_long)
    print(f"共同样本过滤前总数: {len(full_universe)}")
    print(f"共同样本过滤后总数: {len(tradeable_universe)}")
    print(f"共同样本剔除数: {len(full_universe) - len(tradeable_universe)}")

    full_universe.to_csv(
        os.path.join(OUT_DIR, "full_common_universe_before_filter.csv"),
        index=False, encoding="utf-8-sig"
    )
    tradeable_universe.to_csv(
        os.path.join(OUT_DIR, "tradeable_common_universe_after_filter.csv"),
        index=False, encoding="utf-8-sig"
    )

    print("\n=== Step 6: 分别做 common-sample 回测 ===")
    summary_list = []
    equity_map = {}

    for k in TOPK_LIST:
        print(f"Backtesting common-sample Top-{k} ...")

        # RealGraph
        df_daily_real, df_trades_real = backtest_filter_first_topk_common(
            tradeable_universe, all_pred_dates, k, model="RealGraph"
        )
        df_daily_real.to_csv(
            os.path.join(OUT_DIR, f"bt_daily_common_real_K{k}.csv"),
            index=False, encoding="utf-8-sig"
        )
        df_trades_real.to_csv(
            os.path.join(OUT_DIR, f"bt_trades_common_real_K{k}.csv"),
            index=False, encoding="utf-8-sig"
        )

        summ_real = summarize_backtest(df_daily_real)
        summ_real.update({
            "Model": "RealGraph",
            "K": k,
            "BuyCost": BUY_COST,
            "SellCost": SELL_COST,
            "RoundtripCost": ROUNDTRIP_COST,
            "LimitThreshold": LIMIT_THRESHOLD,
            "AvgHold": float(df_daily_real["n_hold"].replace(0, np.nan).mean()) if (df_daily_real["n_hold"] > 0).any() else 0.0,
        })
        summary_list.append(summ_real)
        equity_map[("RealGraph", k)] = df_daily_real

        # ZeroGraph
        df_daily_zero, df_trades_zero = backtest_filter_first_topk_common(
            tradeable_universe, all_pred_dates, k, model="ZeroGraph"
        )
        df_daily_zero.to_csv(
            os.path.join(OUT_DIR, f"bt_daily_common_zero_K{k}.csv"),
            index=False, encoding="utf-8-sig"
        )
        df_trades_zero.to_csv(
            os.path.join(OUT_DIR, f"bt_trades_common_zero_K{k}.csv"),
            index=False, encoding="utf-8-sig"
        )

        summ_zero = summarize_backtest(df_daily_zero)
        summ_zero.update({
            "Model": "ZeroGraph",
            "K": k,
            "BuyCost": BUY_COST,
            "SellCost": SELL_COST,
            "RoundtripCost": ROUNDTRIP_COST,
            "LimitThreshold": LIMIT_THRESHOLD,
            "AvgHold": float(df_daily_zero["n_hold"].replace(0, np.nan).mean()) if (df_daily_zero["n_hold"] > 0).any() else 0.0,
        })
        summary_list.append(summ_zero)
        equity_map[("ZeroGraph", k)] = df_daily_zero

        plot_equity_compare(
            d_real=df_daily_real,
            d_zero=df_daily_zero,
            out_png=os.path.join(OUT_DIR, f"equity_compare_K{k}.png"),
            title=f"Equity Curve | Common Sample | K={k}"
        )

    print("\n=== Step 7: 汇总表 ===")
    df_sum = pd.DataFrame(summary_list).sort_values(["K", "Model"]).reset_index(drop=True)
    df_sum.to_csv(
        os.path.join(OUT_DIR, "bt_summary_common_sample_compare.csv"),
        index=False, encoding="utf-8-sig"
    )

    print("\n========== Common-Sample Backtest Summary ==========")
    print(format_summary_for_display(df_sum).to_string(index=False))

    print(f"\n结果已保存到: {OUT_DIR}")
    print("重点先看：")
    print("1) bt_summary_common_sample_compare.csv")
    print("2) equity_compare_K1.png")
    print("3) equity_compare_K3.png")
    print("4) equity_compare_K5.png")
    print("5) equity_compare_K10.png")


if __name__ == "__main__":
    run_all()