# -*- coding: utf-8 -*-
# quarterly_lasso_daily15_unroll_gapup.py

import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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

# LASSO / time-ordered CV settings
LASSO_MAX_ITER = 50000
CV_FOLDS = 3
N_ALPHAS = 100
ALPHA_MIN_RATIO = 1e-3
COEF_EPS = 1e-10

# 每季度最低样本数（跟你对比代码一致）
MIN_SAMPLES_PER_QUARTER = 5000

# 输出 pkl：保存到“脚本同目录”
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PKL = os.path.join(SCRIPT_DIR, "quarterly_lasso_train_daily15_unroll.pkl")
OUT_COEF_CSV = os.path.join(SCRIPT_DIR, "quarterly_lasso_1se_coefficients.csv")
OUT_SUMMARY_CSV = os.path.join(SCRIPT_DIR, "quarterly_lasso_1se_summary.csv")


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
# 6) Quarterly time-ordered CV + 1-SE rule
# =======================
def make_time_ordered_splits(dates: pd.Series, n_splits: int = CV_FOLDS):
    """
    Build expanding-window CV folds on unique trading dates.

    Every stock observed on the same trading date is assigned to the same
    fold. Consequently, each validation block is strictly later than its
    corresponding training block.
    """
    date_values = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    unique_dates = np.unique(date_values)
    if len(unique_dates) <= n_splits:
        raise RuntimeError(
            f"Need more than {n_splits} unique dates, got {len(unique_dates)}."
        )

    splitter = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    fold_info = []
    for fold, (train_date_pos, val_date_pos) in enumerate(
        splitter.split(unique_dates), start=1
    ):
        train_dates = unique_dates[train_date_pos]
        val_dates = unique_dates[val_date_pos]
        train_idx = np.flatnonzero(np.isin(date_values, train_dates))
        val_idx = np.flatnonzero(np.isin(date_values, val_dates))
        if train_idx.size == 0 or val_idx.size == 0:
            raise RuntimeError(f"Empty observations in time-ordered fold {fold}.")
        if train_dates.max() >= val_dates.min():
            raise RuntimeError(f"Fold {fold} is not strictly time ordered.")
        splits.append((train_idx, val_idx))
        fold_info.append({
            "fold": fold,
            "train_start": str(pd.Timestamp(train_dates.min()).date()),
            "train_end": str(pd.Timestamp(train_dates.max()).date()),
            "validation_start": str(pd.Timestamp(val_dates.min()).date()),
            "validation_end": str(pd.Timestamp(val_dates.max()).date()),
            "n_train": int(train_idx.size),
            "n_validation": int(val_idx.size),
        })
    return splits, fold_info


def build_alpha_grid(X: np.ndarray, y: np.ndarray, splits) -> np.ndarray:
    """Construct one descending alpha grid using training folds only."""
    fold_alpha_max = []
    for train_idx, _ in splits:
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train = imp.fit_transform(X[train_idx])
        X_train = scaler.fit_transform(X_train)
        y_train = y[train_idx]
        y_centered = y_train - np.mean(y_train)
        alpha_max = np.max(np.abs(X_train.T @ y_centered)) / len(y_train)
        if np.isfinite(alpha_max) and alpha_max > 0:
            fold_alpha_max.append(float(alpha_max))
    if not fold_alpha_max:
        raise RuntimeError("Unable to construct a positive LASSO alpha grid.")
    alpha_max = max(fold_alpha_max)
    return np.geomspace(
        alpha_max,
        alpha_max * ALPHA_MIN_RATIO,
        num=N_ALPHAS,
    )


def select_alpha_1se(search: GridSearchCV, n_folds: int) -> dict:
    """Apply the one-standard-error rule to fold-level validation MSEs."""
    cv_results = search.cv_results_
    alphas = np.asarray(
        [params["lasso__alpha"] for params in cv_results["params"]],
        dtype=float,
    )
    mse_path = np.column_stack([
        -np.asarray(cv_results[f"split{k}_test_score"], dtype=float)
        for k in range(n_folds)
    ])

    mean_mse = mse_path.mean(axis=1)
    if n_folds < 2:
        raise RuntimeError("The 1-SE rule requires at least two CV folds.")
    se_mse = mse_path.std(axis=1, ddof=1) / np.sqrt(n_folds)

    min_idx = int(np.argmin(mean_mse))
    threshold = float(mean_mse[min_idx] + se_mse[min_idx])
    eligible = np.flatnonzero(mean_mse <= threshold + 1e-15)
    if eligible.size == 0:
        raise RuntimeError("No alpha satisfies the 1-SE threshold.")

    # Select the largest alpha: the most regularized eligible specification.
    one_se_idx = int(eligible[np.argmax(alphas[eligible])])
    return {
        "alpha_min": float(alphas[min_idx]),
        "alpha_1se": float(alphas[one_se_idx]),
        "cv_mse_min": float(mean_mse[min_idx]),
        "cv_se_at_min": float(se_mse[min_idx]),
        "cv_1se_threshold": threshold,
        "one_se_index": one_se_idx,
    }


def run_quarterly_lasso(panel: pd.DataFrame, feature_cols: list):
    fs = panel[(panel["date"] >= pd.to_datetime(FS_START)) & (panel["date"] <= pd.to_datetime(FS_END))].copy()
    fs["quarter"] = fs["date"].dt.to_period("Q")
    quarters = sorted(fs["quarter"].unique())

    expected_quarters = list(pd.period_range("2022Q2", "2024Q2", freq="Q"))
    if quarters != expected_quarters:
        raise RuntimeError(
            f"Expected the nine quarters 2022Q2-2024Q2, got {quarters}."
        )

    print("\n🔥 参与回归季度：", quarters)

    results = {}
    for q in tqdm(quarters, desc="按季度 time-ordered LASSO"):
        sub = fs[fs["quarter"] == q].dropna(subset=feature_cols, how="all")
        if len(sub) < MIN_SAMPLES_PER_QUARTER:
            print(f"{q} 样本太少（{len(sub)}），跳过")
            continue

        X = sub[feature_cols].values
        y = sub["y"].values
        cv_splits, cv_fold_info = make_time_ordered_splits(
            sub["date"], n_splits=CV_FOLDS
        )
        alphas = build_alpha_grid(X, y, cv_splits)

        # GridSearchCV refits this complete pipeline inside every fold, so
        # median imputation and z-score parameters are learned only from the
        # corresponding training block.
        cv_pipeline = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lasso", Lasso(max_iter=LASSO_MAX_ITER)),
        ])
        search = GridSearchCV(
            estimator=cv_pipeline,
            param_grid={"lasso__alpha": alphas.tolist()},
            scoring="neg_mean_squared_error",
            cv=cv_splits,
            refit=False,
            n_jobs=-1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X, y)
        selection = select_alpha_1se(search, n_folds=CV_FOLDS)
        alpha_1se = selection["alpha_1se"]

        # After model selection, refit on the full quarter at alpha_1se.
        final_pipeline = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lasso", Lasso(
                alpha=alpha_1se,
                max_iter=LASSO_MAX_ITER,
            )),
        ])
        final_pipeline.fit(X, y)
        lasso_1se = final_pipeline.named_steps["lasso"]

        y_hat = final_pipeline.predict(X)
        mse = float(mean_squared_error(y, y_hat))
        r2  = float(r2_score(y, y_hat))

        coef_df = pd.DataFrame({
            "quarter": str(q),
            "feature": feature_cols,
            "coef": lasso_1se.coef_,
            "active": np.abs(lasso_1se.coef_) > COEF_EPS,
            "coef_epsilon": COEF_EPS,
            "alpha_min": selection["alpha_min"],
            "alpha_1se": alpha_1se,
        })

        results[str(q)] = {
            "alpha": alpha_1se,
            "alpha_min": selection["alpha_min"],
            "alpha_1se": alpha_1se,
            "cv_mse_min": selection["cv_mse_min"],
            "cv_se_at_min": selection["cv_se_at_min"],
            "cv_1se_threshold": selection["cv_1se_threshold"],
            "cv_folds": cv_fold_info,
            "mse": mse,
            "r2": r2,
            "coef_table": coef_df
        }

        print(
            f"{q} ✔ alpha_min={selection['alpha_min']:.3e}, "
            f"alpha_1se={alpha_1se:.3e}, R²={r2:.4f}, "
            f"active={np.sum(np.abs(lasso_1se.coef_) > COEF_EPS)}"
        )

    return results


def main():
    panel, feats = build_panel()
    results = run_quarterly_lasso(panel, feats)

    with open(OUT_PKL, "wb") as f:
        pickle.dump(results, f)

    coef_tables = [item["coef_table"] for item in results.values()]
    if not coef_tables:
        raise RuntimeError("No quarterly coefficient tables were produced.")
    pd.concat(coef_tables, ignore_index=True).to_csv(
        OUT_COEF_CSV, index=False, encoding="utf-8-sig"
    )

    summary_rows = []
    for quarter, item in results.items():
        summary_rows.append({
            "quarter": quarter,
            "alpha_min": item["alpha_min"],
            "alpha_1se": item["alpha_1se"],
            "cv_mse_min": item["cv_mse_min"],
            "cv_se_at_min": item["cv_se_at_min"],
            "cv_1se_threshold": item["cv_1se_threshold"],
            "in_sample_mse": item["mse"],
            "in_sample_r2": item["r2"],
            "active_features": int(item["coef_table"]["active"].sum()),
        })
    pd.DataFrame(summary_rows).to_csv(
        OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig"
    )

    print(f"\n✨ 已保存：{OUT_PKL}")
    print(f"✨ 已保存：{OUT_COEF_CSV}")
    print(f"✨ 已保存：{OUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
