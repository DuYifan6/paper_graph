# -*- coding: utf-8 -*-
# quarterly_lasso_daily15_unroll_gapup.py

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


# =======================
# 0) 配置（只改这里）
# =======================
BASE_DIR = r"D:\Lasso\归一化数据"

DAILY_DIR  = os.path.join(BASE_DIR, r"train\daily")
MINUTE_DIR = os.path.join(BASE_DIR, r"train\minute")

FS_START = "2022-04-01"
FS_END   = "2024-06-28"

Y_COL = "gap_up_pct"          # 只用这一列作为 y
MINUTE_SUFFIX = "_整合.csv"   # 000001.SZ.csv -> 000001.SZ_整合.csv

# 你的 15min 每天只有 5 根（13:30~14:30）
N_BARS = 5

# t-1 预测 t
LAG_DAYS = 1

# LassoCV 迭代上限（降低 ConvergenceWarning）
LASSO_MAX_ITER = 50000

# 每季度最低样本数（跟你对比代码一致）
MIN_SAMPLES_PER_QUARTER = 5000

# 输出 pkl：保存到“脚本同目录”
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PKL = os.path.join(SCRIPT_DIR, "quarterly_lasso_train_daily15_unroll.pkl")


# =======================
# 1) 工具：匹配 daily 与 minute 文件
# =======================
def list_stock_pairs(daily_dir: str, minute_dir: str, minute_suffix: str):
    """
    以 DAILY_DIR 下 *.csv 为准，匹配 MINUTE_DIR 下 <code>_整合.csv
    """
    daily_files = sorted(glob.glob(os.path.join(daily_dir, "*.csv")))
    pairs = []
    for dpath in daily_files:
        code = os.path.basename(dpath).replace(".csv", "")
        mpath = os.path.join(minute_dir, f"{code}{minute_suffix}")
        if os.path.exists(mpath):
            pairs.append((code, dpath, mpath))
    return pairs


# =======================
# 2) 读日频：固定 y=gap_up_pct，且剔除“末尾两列标签”中的另一列
# =======================
def load_daily_one(daily_csv: str) -> pd.DataFrame:
    df = pd.read_csv(daily_csv)

    if "trade_date" not in df.columns:
        raise ValueError(f"{daily_csv} 缺少 trade_date 列")

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()

    if "ts_code" not in df.columns:
        code = os.path.basename(daily_csv).replace(".csv", "")
        df["ts_code"] = code

    if Y_COL not in df.columns:
        raise ValueError(f"{daily_csv} 缺少 y 列 {Y_COL}")

    # ✅ 关键：无条件剔除 gap_up_flag（不管它在哪一列）
    DROP_LABELS = {"gap_up_flag"}
    drop_label_cols = [c for c in df.columns if c in DROP_LABELS]

    # y
    df["y"] = pd.to_numeric(df[Y_COL], errors="coerce")

    # 日频特征加前缀 D_
    exclude = {"ts_code", "trade_date", "y", Y_COL} | set(drop_label_cols)
    feat_cols = [c for c in df.columns if c not in exclude]

    # 强制转数值
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")

    df = df.rename(columns={c: "D_" + c for c in feat_cols})

    keep = ["ts_code", "trade_date", "y"] + ["D_" + c for c in feat_cols]
    out = df[keep].dropna(subset=["trade_date", "y"]).reset_index(drop=True)
    return out



# =======================
# 3) 读 15min：按天展开成 M_{feat}_{k}（k=1..N_BARS）
# =======================
def load_minute_wide_one(minute_csv: str) -> pd.DataFrame:
    df = pd.read_csv(minute_csv)

    if "trade_time" not in df.columns:
        raise ValueError(f"{minute_csv} 缺少 trade_time 列")

    df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
    df["trade_date"] = df["trade_time"].dt.normalize()

    if "ts_code" not in df.columns:
        # 兜底：用文件名推 ts_code（000001.SZ_整合.csv）
        base = os.path.basename(minute_csv)
        df["ts_code"] = base.split("_")[0].replace(".csv", "")

    # 识别分钟特征列（排除 id/time/date）
    exclude = {"ts_code", "trade_time", "trade_date"}
    minute_feats = [c for c in df.columns if c not in exclude]

    # 强制转数值（避免 dtype=object 直接被丢）
    df[minute_feats] = df[minute_feats].apply(pd.to_numeric, errors="coerce")

    # 排序、编号 bar_idx
    df = df.sort_values(["ts_code", "trade_date", "trade_time"])
    df["bar_idx"] = df.groupby(["ts_code", "trade_date"], sort=False).cumcount()

    # 只取前 N_BARS 根（你这里就是 5 根）
    df = df[df["bar_idx"] < N_BARS].copy()
    df["bar"] = df["bar_idx"] + 1  # 1..N_BARS

    # pivot：每个 feat -> N_BARS 列
    wide_parts = []
    for feat in minute_feats:
        p = df.pivot_table(index=["ts_code", "trade_date"], columns="bar", values=feat, aggfunc="first")
        p.columns = [f"M_{feat}_{int(k)}" for k in p.columns]
        wide_parts.append(p)

    wide = pd.concat(wide_parts, axis=1).reset_index()
    return wide


# =======================
# 4) 每只股票合并 + 做 t-1 -> t（关键：避免全量 panel shift 的内存炸裂）
# =======================
def apply_lag_one_stock(merged: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    merged = merged.sort_values("trade_date").reset_index(drop=True)
    merged["date"] = pd.to_datetime(merged["trade_date"], errors="coerce")

    exclude = {"ts_code", "trade_date", "date", "y"}
    feat_cols = [c for c in merged.columns if c not in exclude]

    # ✅ 关键：先把特征列转成 float（能容纳 NaN），顺便省内存用 float32
    merged[feat_cols] = merged[feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)

    # shift 特征
    merged[feat_cols] = merged[feat_cols].shift(lag_days)

    # 丢掉前 lag_days 行（这些行没有 t-1 特征）
    if len(merged) <= lag_days:
        return merged.iloc[0:0].copy()
    merged = merged.iloc[lag_days:].copy()

    return merged



# =======================
# 5) 构建 panel（按股票逐个处理，降低峰值内存）
# =======================
def build_panel():
    pairs = list_stock_pairs(DAILY_DIR, MINUTE_DIR, MINUTE_SUFFIX)
    if not pairs:
        raise RuntimeError("❌ 没找到可匹配的 (daily, minute) 文件对，请检查目录和 MINUTE_SUFFIX。")

    panels = []
    buf = []

    for code, dpath, mpath in tqdm(pairs, desc="读取&拼接 daily+15min"):
        daily = load_daily_one(dpath)
        minute_wide = load_minute_wide_one(mpath)

        merged = daily.merge(minute_wide, on=["ts_code", "trade_date"], how="left")

        # t-1 -> t
        merged = apply_lag_one_stock(merged, lag_days=LAG_DAYS)
        if merged.empty:
            continue

        buf.append(merged)

        # 分批 concat，减少小对象堆积
        if len(buf) >= 200:
            panels.append(pd.concat(buf, ignore_index=True))
            buf = []

    if buf:
        panels.append(pd.concat(buf, ignore_index=True))

    panel = pd.concat(panels, ignore_index=True)

    # 严格按时间排序
    panel = panel.sort_values(["ts_code", "date"]).reset_index(drop=True)

    # feature_cols：和你对比代码一样的定义方式（排除 id/date/y）
    exclude = {"ts_code", "trade_date", "date", "y"}
    feature_cols = [c for c in panel.columns if c not in exclude]

    # 再次确保都是数值（避免 object 混进来）
    panel[feature_cols] = panel[feature_cols].apply(pd.to_numeric, errors="coerce")

    # 删除全空列（避免某些分钟列完全缺失）
    feature_cols = [c for c in feature_cols if not panel[c].isna().all()]

    print(f"[INFO] panel shape={panel.shape} | dates={panel['date'].nunique()} | stocks={panel['ts_code'].nunique()}")
    print(f"[INFO] feature_cols={len(feature_cols)}")

    return panel, feature_cols


# =======================
# 6) 按季度 LassoCV（输出结构与对比代码一致）
# =======================
def run_quarterly_lasso(panel: pd.DataFrame, feature_cols: list):
    fs = panel[(panel["date"] >= pd.to_datetime(FS_START)) & (panel["date"] <= pd.to_datetime(FS_END))].copy()
    fs["quarter"] = fs["date"].dt.to_period("Q")
    quarters = sorted(fs["quarter"].unique())

    print("\n🔥 参与回归季度：", quarters)

    results = {}
    for q in tqdm(quarters, desc="按季度 LassoCV"):
        sub = fs[fs["quarter"] == q].dropna(subset=feature_cols, how="all")
        if len(sub) < MIN_SAMPLES_PER_QUARTER:
            print(f"{q} 样本太少（{len(sub)}），跳过")
            continue

        X = sub[feature_cols].values
        y = sub["y"].values

        # Providing the SAME pkl content structure as your baseline code:
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lasso", LassoCV(cv=3, n_jobs=-1, random_state=0, max_iter=LASSO_MAX_ITER))
        ])
        pipe.fit(X, y)

        lasso = pipe.named_steps["lasso"]
        alpha = float(lasso.alpha_)

        y_hat = pipe.predict(X)
        mse = float(mean_squared_error(y, y_hat))
        r2  = float(r2_score(y, y_hat))

        coef_df = pd.DataFrame({"feature": feature_cols, "coef": lasso.coef_})

        # ✅ pkl 内容结构：与“你对比的那个代码”完全一致
        results[str(q)] = {
            "alpha": alpha,
            "mse": mse,
            "r2": r2,
            "coef_table": coef_df
        }

        print(f"{q} ✔ alpha={alpha:.3e}, R²={r2:.4f}, 非零={np.sum(lasso.coef_!=0)}")

    return results


def main():
    panel, feats = build_panel()
    results = run_quarterly_lasso(panel, feats)

    with open(OUT_PKL, "wb") as f:
        pickle.dump(results, f)

    print(f"\n✨ 已保存：{OUT_PKL}")


if __name__ == "__main__":
    main()
