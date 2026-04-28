# -*- coding: utf-8 -*-
# lasso_quarterly_train_only_daily15_unroll.py

import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score




DAILY_DIR = r"D:\Lasso\归一化数据\merged\daily"

MIN15_DIR = r"D:\Lasso\归一化数据\merged\minute"

# =========================================================
# 关键修改：只在训练集上做特征筛选
# =========================================================
TRAIN_START = "2022-01-01"
TRAIN_END   = "2024-12-31"

# 如果你只是想读入全样本但筛选只用训练集，保持 False
# 如果你想在构建 panel 时也直接裁剪到训练集，可以改 True
FILTER_PANEL_TO_TRAIN_ONLY = False

# 15min 每天 5 根 K 线
N_BARS = 5

# 输出
OUT_DIR = r"D:\PythonProject\LASSO_FINAL\lasso_feature_selection_train_only"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PKL = os.path.join(OUT_DIR, "quarterly_lasso_train_only_daily15_unroll.pkl")
OUT_SELECTED_EACH_Q_CSV = os.path.join(OUT_DIR, "selected_features_by_quarter_train_only.csv")
OUT_STABILITY_CSV = os.path.join(OUT_DIR, "selected_features_stability_train_only.csv")
OUT_PANEL_INFO_CSV = os.path.join(OUT_DIR, "panel_info_train_only.csv")


# =========================================================
# 1) 读日频：y 取最后一列
# =========================================================
def load_daily_with_y():
    files = glob.glob(os.path.join(DAILY_DIR, "*.csv"))
    print(f"📌 daily 文件数：{len(files)}")

    dfs = []
    for f in tqdm(files, desc="读取 daily"):
        df = pd.read_csv(f)

        if "trade_date" not in df.columns:
            raise ValueError(f"{f} 缺少 trade_date 列")
        if "ts_code" not in df.columns:
            raise ValueError(f"{f} 缺少 ts_code 列")

        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["trade_date"]).copy()

        # 最后一列作为连续 y
        y_col = df.columns[-1]
        df = df.rename(columns={y_col: "y"})

        exclude = {"ts_code", "trade_date", "y"}
        feat_cols = [c for c in df.columns if c not in exclude]

        # 给日频特征加 D_ 前缀
        rename_map = {c: "D_" + c for c in feat_cols}
        df = df.rename(columns=rename_map)

        keep = ["ts_code", "trade_date", "y"] + list(rename_map.values())
        dfs.append(df[keep])

    if not dfs:
        raise RuntimeError("❌ daily 没读到任何可用文件")

    return pd.concat(dfs, ignore_index=True)


# =========================================================
# 2) 读 15min：不聚合，按天展开成 M_{feat}_{k}
# =========================================================
def load_minute_unrolled():
    files = glob.glob(os.path.join(MIN15_DIR, "*.csv"))
    print(f"\n📌 minute 文件数：{len(files)}")

    out_list = []
    for f in tqdm(files, desc="读取 minute"):
        df = pd.read_csv(f)

        if "trade_time" not in df.columns:
            raise ValueError(f"{f} 缺少 trade_time 列")

        df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
        df = df.dropna(subset=["trade_time"]).copy()
        df["trade_date"] = df["trade_time"].dt.strftime("%Y-%m-%d")

        if "ts_code" not in df.columns:
            base = os.path.basename(f)
            df["ts_code"] = base.split("_")[0].replace(".csv", "")

        exclude = {"ts_code", "trade_time", "trade_date"}
        minute_feats = [c for c in df.columns if c not in exclude]
        minute_feats = [c for c in minute_feats if pd.api.types.is_numeric_dtype(df[c])]

        if not minute_feats:
            continue

        df = df.sort_values(["ts_code", "trade_date", "trade_time"])
        df["bar_idx"] = df.groupby(["ts_code", "trade_date"]).cumcount()
        df = df[df["bar_idx"] < N_BARS].copy()

        wide_rows = []
        for (ts, d), sub in df.groupby(["ts_code", "trade_date"], sort=False):
            sub = sub.sort_values("trade_time")
            row = {"ts_code": ts, "trade_date": d}

            for k in range(N_BARS):
                if k < len(sub):
                    for feat in minute_feats:
                        row[f"M_{feat}_{k+1}"] = sub.iloc[k][feat]
                else:
                    for feat in minute_feats:
                        row[f"M_{feat}_{k+1}"] = np.nan

            wide_rows.append(row)

        if wide_rows:
            out_list.append(pd.DataFrame(wide_rows))

    if not out_list:
        raise RuntimeError("❌ minute 没读到任何可用数据")

    return pd.concat(out_list, ignore_index=True)


# =========================================================
# 3) 构建 panel + shift(1)
# =========================================================
def build_panel():
    daily = load_daily_with_y()
    minute_wide = load_minute_unrolled()

    print("\n📌 合并日频 + 15min 展开")
    panel = daily.merge(minute_wide, on=["ts_code", "trade_date"], how="left")

    panel["date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).copy()
    panel = panel.sort_values(["ts_code", "date"])

    panel = panel.dropna(subset=["y"]).copy()

    exclude = {"ts_code", "trade_date", "date", "y"}
    feature_cols = [c for c in panel.columns if c not in exclude]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(panel[c])]

    print(f"📌 合并后特征数：{len(feature_cols)}")
    print(f"📌 panel 行数 shift 前：{len(panel):,}")

    # t-1 特征预测 t 的 y
    panel[feature_cols] = panel.groupby("ts_code", group_keys=False)[feature_cols].shift(1)

    # 可选：构建阶段就只保留训练集
    if FILTER_PANEL_TO_TRAIN_ONLY:
        train_start = pd.to_datetime(TRAIN_START)
        train_end = pd.to_datetime(TRAIN_END)
        panel = panel[(panel["date"] >= train_start) & (panel["date"] <= train_end)].copy()

    # y 不能缺
    panel = panel.dropna(subset=["y"]).copy()

    # 注意：这里先不按全样本删除全空列，避免用测试期信息判断列是否有效
    print(f"📌 panel 行数 shift 后：{len(panel):,}")
    print(f"📌 panel 日期范围：{panel['date'].min().date()} ~ {panel['date'].max().date()}")

    return panel, feature_cols


# =========================================================
# 4) 只在训练集上确定有效特征列
# =========================================================
def get_train_feature_cols(panel, feature_cols):
    train_start = pd.to_datetime(TRAIN_START)
    train_end = pd.to_datetime(TRAIN_END)

    train_panel = panel[(panel["date"] >= train_start) & (panel["date"] <= train_end)].copy()

    if train_panel.empty:
        raise ValueError(
            f"❌ 训练集为空，请检查 TRAIN_START/TRAIN_END: {TRAIN_START} ~ {TRAIN_END}"
        )

    # 只用训练集判断哪些特征不是全空
    train_feature_cols = [c for c in feature_cols if not train_panel[c].isna().all()]

    print("\n========== 训练集筛选范围 ==========")
    print(f"TRAIN_START: {TRAIN_START}")
    print(f"TRAIN_END  : {TRAIN_END}")
    print(f"训练集行数  : {len(train_panel):,}")
    print(f"训练集股票数: {train_panel['ts_code'].nunique():,}")
    print(f"训练集日期  : {train_panel['date'].min().date()} ~ {train_panel['date'].max().date()}")
    print(f"训练集有效特征数: {len(train_feature_cols)}")

    info = pd.DataFrame([{
        "TRAIN_START": TRAIN_START,
        "TRAIN_END": TRAIN_END,
        "n_rows_train": len(train_panel),
        "n_stocks_train": train_panel["ts_code"].nunique(),
        "date_min_train": train_panel["date"].min(),
        "date_max_train": train_panel["date"].max(),
        "n_features_before_train_filter": len(feature_cols),
        "n_features_after_train_filter": len(train_feature_cols),
    }])
    info.to_csv(OUT_PANEL_INFO_CSV, index=False, encoding="utf-8-sig")

    return train_panel, train_feature_cols


# =========================================================
# 5) 按季度 LassoCV：只用训练集季度
# =========================================================
def run_quarterly_lasso_train_only(train_panel, feature_cols):
    fs = train_panel.copy()
    fs["quarter"] = fs["date"].dt.to_period("Q")
    quarters = sorted(fs["quarter"].unique())

    print("\n🔥 只在训练集参与 LassoCV 的季度：")
    print(quarters)

    results = {}
    selected_rows = []

    for q in tqdm(quarters, desc="按训练集季度 LassoCV"):
        sub = fs[fs["quarter"] == q].dropna(subset=feature_cols, how="all").copy()

        if len(sub) < 5000:
            print(f"{q} 样本太少（{len(sub)}），跳过")
            continue

        X = sub[feature_cols].values
        y = pd.to_numeric(sub["y"], errors="coerce").values

        valid_y = np.isfinite(y)
        X = X[valid_y]
        y = y[valid_y]

        if len(y) < 5000:
            print(f"{q} 有效 y 样本太少（{len(y)}），跳过")
            continue

        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lasso", LassoCV(
                cv=3,
                n_jobs=-1,
                random_state=0,
                max_iter=20000
            ))
        ])

        pipe.fit(X, y)

        lasso = pipe.named_steps["lasso"]
        alpha = float(lasso.alpha_)

        y_hat = pipe.predict(X)
        mse = mean_squared_error(y, y_hat)
        r2 = r2_score(y, y_hat)

        coef = lasso.coef_
        coef_df = pd.DataFrame({
            "feature": feature_cols,
            "coef": coef,
            "abs_coef": np.abs(coef),
            "selected": coef != 0
        }).sort_values("abs_coef", ascending=False)

        selected_df = coef_df[coef_df["selected"]].copy()
        selected_df["quarter"] = str(q)
        selected_df["alpha"] = alpha
        selected_df["mse"] = mse
        selected_df["r2"] = r2
        selected_df["n_obs"] = len(y)

        selected_rows.append(selected_df)

        results[str(q)] = {
            "alpha": alpha,
            "mse": mse,
            "r2": r2,
            "n_obs": len(y),
            "n_selected": int((coef != 0).sum()),
            "coef_table": coef_df,
        }

        print(
            f"{q} ✔ alpha={alpha:.3e}, "
            f"R²={r2:.4f}, "
            f"非零={int((coef != 0).sum())}, "
            f"n={len(y):,}"
        )

    # 保存每季度选中特征
    if selected_rows:
        selected_all = pd.concat(selected_rows, ignore_index=True)
    else:
        selected_all = pd.DataFrame(columns=[
            "feature", "coef", "abs_coef", "selected",
            "quarter", "alpha", "mse", "r2", "n_obs"
        ])

    selected_all.to_csv(OUT_SELECTED_EACH_Q_CSV, index=False, encoding="utf-8-sig")

    # 稳定性统计
    stability = build_stability_table(selected_all, quarters)
    stability.to_csv(OUT_STABILITY_CSV, index=False, encoding="utf-8-sig")

    return results, selected_all, stability


# =========================================================
# 6) 构造跨季度稳定性表
# =========================================================
def build_stability_table(selected_all, quarters):
    if selected_all.empty:
        return pd.DataFrame(columns=[
            "feature", "appear_count", "appear_freq",
            "avg_coef", "avg_abs_coef", "pos_ratio"
        ])

    n_q = len(quarters)

    tmp = selected_all.copy()
    tmp["pos"] = (tmp["coef"] > 0).astype(int)

    stability = (
        tmp.groupby("feature")
        .agg(
            appear_count=("quarter", "nunique"),
            avg_coef=("coef", "mean"),
            avg_abs_coef=("abs_coef", "mean"),
            pos_ratio=("pos", "mean")
        )
        .reset_index()
    )

    stability["appear_freq"] = stability["appear_count"] / n_q
    stability = stability.sort_values(
        ["appear_count", "avg_abs_coef"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return stability


# =========================================================
# 7) 主程序
# =========================================================
def main():
    panel, all_feats = build_panel()

    # 关键：只用训练集确定有效特征列
    train_panel, train_feats = get_train_feature_cols(panel, all_feats)

    # 关键：只在训练集季度上做 LASSO
    results, selected_all, stability = run_quarterly_lasso_train_only(
        train_panel,
        train_feats
    )

    with open(OUT_PKL, "wb") as f:
        pickle.dump(results, f)

    print("\n========== 输出完成 ==========")
    print(f"PKL 结果：{OUT_PKL}")
    print(f"每季度选中特征：{OUT_SELECTED_EACH_Q_CSV}")
    print(f"跨季度稳定性表：{OUT_STABILITY_CSV}")
    print(f"panel 信息：{OUT_PANEL_INFO_CSV}")

    print("\nTop 30 稳定特征：")
    if not stability.empty:
        print(stability.head(30).to_string(index=False))
    else:
        print("无稳定特征。")


if __name__ == "__main__":
    main()
