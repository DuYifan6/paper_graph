#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1: Build final ESWA SOTA benchmark table.

Models:
Ridge, LightGBM, MLP, Transformer, GCN-LSTM, ZeroGraph, Proposed Graph-2.

Outputs:
final_eswa_sota/tables/
    Table3_SOTA.csv
    Table3_SOTA.xlsx
    Table3_SOTA_monthly.csv
    Table3_SOTA_topk.csv
    Table3_SOTA_input_diagnostics.csv

Run:
D:\Anaconda3\python.exe D:\PythonProject\LASSO_FINAL\SOTA\build_final_sota_table.py
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# =============================================================================
# 0) PATH CONFIGURATION
# =============================================================================

SOTA_RESULTS_REP = r"D:\PythonProject\LASSO_FINAL\SOTA\sota_results_rep"
GCN_LSTM_RESULTS = r"D:\PythonProject\LASSO_FINAL\SOTA\sota_results_gcn_lstm"

ZEROGRAPH_TEST_DETAIL = r"D:\PythonProject\LASSO_FINAL\graph_调整\fixed_zerograph_valselect\results\predictions_ZeroGraph_test_detail.csv"
GRAPH2_TEST_DETAIL = r"D:\PythonProject\LASSO_FINAL\graph_调整\graph2_binary_60\results\predictions_Graph2_C_K3510_test_detail.csv"

OUT_DIR = r"D:\PythonProject\LASSO_FINAL\SOTA\final_eswa_sota"

REPRESENTATIVE_MODELS = {
    "ridge": {"display": "Ridge", "category": "Regularized linear model", "file_stem": "pred_ridge"},
    "lightgbm": {"display": "LightGBM", "category": "Tree ensemble model", "file_stem": "pred_lightgbm"},
    "mlp": {"display": "MLP", "category": "Neural network", "file_stem": "pred_mlp"},
    "transformer": {"display": "Transformer", "category": "Temporal deep model", "file_stem": "pred_transformer"},
}

OTHER_MODELS = {
    "gcn_lstm": {"display": "GCN-LSTM", "category": "Graph neural network", "path": os.path.join(GCN_LSTM_RESULTS, "pred_gcn_lstm.csv")},
    "zerograph": {"display": "ZeroGraph", "category": "No-graph temporal backbone", "path": ZEROGRAPH_TEST_DETAIL},
    "graph2": {"display": "Proposed Graph-2", "category": "Graph-as-Regularizer", "path": GRAPH2_TEST_DETAIL},
}

TOPK_LIST = [1, 3, 5, 10]


# =============================================================================
# 1) IO AND STANDARDIZATION
# =============================================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_prediction_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(path)
    if suffix in [".pkl", ".pickle"]:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported prediction file type: {path}")


def find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def normalize_date_col(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.number):
        return pd.to_datetime(s.astype(int).astype(str), errors="coerce").dt.strftime("%Y-%m-%d")
    return pd.to_datetime(s.astype(str), errors="coerce").dt.strftime("%Y-%m-%d")


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def standardize_prediction_df(raw: pd.DataFrame, model_name: str, path: str | Path) -> pd.DataFrame:
    df = raw.copy()
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed:")]]

    date_col = find_col(df, ["date", "trade_date", "datetime", "cal_date", "day"])
    stock_col = find_col(df, ["stock", "ts_code", "code", "ticker", "symbol", "n_idx", "stock_id", "node", "node_id"])
    y_true_col = find_col(df, [
        "y_true", "true", "actual", "real", "realized", "label", "target", "y",
        "ret", "return", "gap", "gap_up_pct", "overnight_gap_return",
        "overnight_return", "y_real", "y_test"
    ])
    y_pred_col = find_col(df, [
        "y_pred", "pred", "prediction", "predict", "forecast", "yhat", "y_hat",
        "pred_ret", "pred_return", "score", "pred_gap"
    ])

    if y_pred_col is None:
        pred_like = [c for c in df.columns if "pred" in str(c).lower() or "forecast" in str(c).lower()]
        if len(pred_like) == 1:
            y_pred_col = pred_like[0]

    if y_true_col is None:
        true_like = [c for c in df.columns if any(x in str(c).lower() for x in ["true", "actual", "real", "target", "label"])]
        if len(true_like) == 1:
            y_true_col = true_like[0]

    missing = []
    if date_col is None:
        missing.append("date")
    if stock_col is None:
        missing.append("stock")
    if y_true_col is None:
        missing.append("y_true")
    if y_pred_col is None:
        missing.append("y_pred")
    if missing:
        raise KeyError(
            f"Cannot standardize {model_name} from {path}. Missing {missing}.\n"
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "date": normalize_date_col(df[date_col]),
        "stock": df[stock_col].astype(str),
        "y_true": pd.to_numeric(df[y_true_col], errors="coerce"),
        "y_pred": pd.to_numeric(df[y_pred_col], errors="coerce"),
        "model": model_name,
    })

    out = out.replace([np.inf, -np.inf], np.nan)
    before = len(out)
    out = out.dropna(subset=["date", "stock", "y_true", "y_pred"])
    if len(out) < before:
        print(f"[{model_name}] dropped {before - len(out):,} invalid rows.")

    dup = out.duplicated(["date", "stock"]).sum()
    if dup:
        print(f"[{model_name}] warning: {dup:,} duplicated date-stock rows; keeping last.")
        out = out.drop_duplicates(["date", "stock"], keep="last")

    return out


# =============================================================================
# 2) METRICS
# =============================================================================

def r2_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) < 2:
        return np.nan
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.nan if sst <= 1e-12 else float(1.0 - sse / sst)


def mse_np(y_true, y_pred) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae_np(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse_np(y_true, y_pred) -> float:
    return float(np.sqrt(mse_np(y_true, y_pred)))


def daily_corr_metrics(df: pd.DataFrame) -> Tuple[float, float, float, float, int]:
    ic_vals, rank_vals = [], []
    for _, g in df.groupby("date", sort=True):
        if len(g) < 10:
            continue
        y = g["y_true"].to_numpy(dtype=float)
        p = g["y_pred"].to_numpy(dtype=float)

        if np.nanstd(y) > 1e-12 and np.nanstd(p) > 1e-12:
            ic = np.corrcoef(y, p)[0, 1]
            if np.isfinite(ic):
                ic_vals.append(float(ic))

        if pd.Series(y).nunique() >= 3 and pd.Series(p).nunique() >= 3:
            ric = spearmanr(y, p).correlation
            if np.isfinite(ric):
                rank_vals.append(float(ric))

    def mean_se(vals):
        if not vals:
            return np.nan, np.nan
        vals = np.asarray(vals, dtype=float)
        return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan

    ic_mean, ic_se = mean_se(ic_vals)
    ric_mean, ric_se = mean_se(rank_vals)
    return ic_mean, ic_se, ric_mean, ric_se, int(df["date"].nunique())


def topk_metrics(df: pd.DataFrame, ks: List[int]) -> pd.DataFrame:
    rows = []
    for k in ks:
        daily = []
        for d, g in df.groupby("date", sort=True):
            top = g.sort_values("y_pred", ascending=False).head(k)
            if len(top) == 0:
                continue
            daily.append({
                "date": d,
                "k": k,
                "topk_mean_y": float(top["y_true"].mean()),
                "topk_median_y": float(top["y_true"].median()),
                "topk_win_rate": float((top["y_true"] > 0).mean()),
                "n": int(len(top)),
            })

        if not daily:
            rows.append({"k": k, "n_days": 0, "topk_mean_y": np.nan, "topk_mean_y_se": np.nan, "topk_win_rate": np.nan, "topk_median_y": np.nan})
            continue

        tmp = pd.DataFrame(daily)
        vals = tmp["topk_mean_y"].to_numpy(dtype=float)
        rows.append({
            "k": k,
            "n_days": int(len(tmp)),
            "topk_mean_y": float(np.mean(vals)),
            "topk_mean_y_se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan,
            "topk_win_rate": float(tmp["topk_win_rate"].mean()),
            "topk_median_y": float(tmp["topk_median_y"].mean()),
        })
    return pd.DataFrame(rows)


def monthly_metrics(df: pd.DataFrame, model_display: str) -> pd.DataFrame:
    d = df.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").astype(str)

    rows = []
    for m, g in d.groupby("month", sort=True):
        ic_mean, _, ric_mean, _, n_days = daily_corr_metrics(g)
        rows.append({
            "model": model_display,
            "month": m,
            "n_obs": int(len(g)),
            "n_days": int(n_days),
            "r2": r2_np(g["y_true"], g["y_pred"]),
            "mse": mse_np(g["y_true"], g["y_pred"]),
            "mae": mae_np(g["y_true"], g["y_pred"]),
            "rmse": rmse_np(g["y_true"], g["y_pred"]),
            "ic": ic_mean,
            "rank_ic": ric_mean,
        })
    return pd.DataFrame(rows)


def summarize_model(df: pd.DataFrame, model_display: str, category: str):
    y = df["y_true"].to_numpy(dtype=float)
    p = df["y_pred"].to_numpy(dtype=float)

    ic_mean, ic_se, ric_mean, ric_se, n_days = daily_corr_metrics(df)
    mon = monthly_metrics(df, model_display)
    tk = topk_metrics(df, TOPK_LIST)

    weighted_monthly_r2 = np.nan
    if len(mon):
        w = mon["n_obs"].to_numpy(dtype=float) / mon["n_obs"].sum()
        weighted_monthly_r2 = float(np.nansum(w * mon["r2"].to_numpy(dtype=float)))

    row = {
        "Model": model_display,
        "Category": category,
        "N_obs": int(len(df)),
        "N_days": int(n_days),
        "OOS_R2_all": r2_np(y, p),
        "Weighted_monthly_R2": weighted_monthly_r2,
        "MSE": mse_np(y, p),
        "MAE": mae_np(y, p),
        "RMSE": rmse_np(y, p),
        "IC": ic_mean,
        "IC_SE": ic_se,
        "RankIC": ric_mean,
        "RankIC_SE": ric_se,
    }

    for _, r in tk.iterrows():
        k = int(r["k"])
        row[f"Top{k}_mean_y"] = r["topk_mean_y"]
        row[f"Top{k}_win_rate"] = r["topk_win_rate"]

    tk.insert(0, "Model", model_display)
    return row, mon, tk


# =============================================================================
# 3) LOAD ALL MODELS
# =============================================================================

def load_rep_model_predictions(model_key: str, cfg: Dict) -> pd.DataFrame:
    root = Path(SOTA_RESULTS_REP)
    stem = cfg["file_stem"]
    possible = [root / f"{stem}.parquet", root / f"{stem}.csv", root / f"{stem}.pkl"]
    path = find_first_existing(possible)
    if path is None:
        raise FileNotFoundError("Cannot find prediction file for " + model_key + ". Tried:\n" + "\n".join(str(p) for p in possible))
    raw = read_prediction_file(path)
    df = standardize_prediction_df(raw, cfg["display"], path)
    print(f"[OK] {cfg['display']}: {len(df):,} rows from {path}")
    return df


def load_other_model_predictions(model_key: str, cfg: Dict) -> pd.DataFrame:
    path = Path(cfg["path"])
    raw = read_prediction_file(path)
    df = standardize_prediction_df(raw, cfg["display"], path)
    print(f"[OK] {cfg['display']}: {len(df):,} rows from {path}")
    return df


def load_all_predictions() -> Dict[str, pd.DataFrame]:
    out = {}
    for k, cfg in REPRESENTATIVE_MODELS.items():
        out[k] = load_rep_model_predictions(k, cfg)
    for k, cfg in OTHER_MODELS.items():
        out[k] = load_other_model_predictions(k, cfg)
    return out


def common_date_stock_diagnostics(preds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows, sets = [], {}
    for k, df in preds.items():
        keyset = set(zip(df["date"].astype(str), df["stock"].astype(str)))
        sets[k] = keyset
        rows.append({
            "model_key": k,
            "n_rows": len(df),
            "n_unique_date_stock": len(keyset),
            "n_dates": df["date"].nunique(),
            "date_start": min(df["date"]),
            "date_end": max(df["date"]),
        })

    common = set.intersection(*sets.values()) if sets else set()
    for r in rows:
        r["n_common_all_models"] = len(common)
        r["coverage_common_ratio"] = len(common) / r["n_unique_date_stock"] if r["n_unique_date_stock"] else np.nan
    return pd.DataFrame(rows)


# =============================================================================
# 4) MAIN
# =============================================================================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    out_dir = ensure_dir(OUT_DIR)
    tables_dir = ensure_dir(out_dir / "tables")

    print("=" * 90)
    print("Building final ESWA SOTA table")
    print("=" * 90)

    preds = load_all_predictions()

    diag = common_date_stock_diagnostics(preds)
    diag.to_csv(tables_dir / "Table3_SOTA_input_diagnostics.csv", index=False, encoding="utf-8-sig")
    print("\nInput diagnostics:")
    print(diag.to_string(index=False))

    summary_rows, monthly_list, topk_list = [], [], []
    model_order = list(REPRESENTATIVE_MODELS.keys()) + list(OTHER_MODELS.keys())

    for k in model_order:
        if k in REPRESENTATIVE_MODELS:
            display = REPRESENTATIVE_MODELS[k]["display"]
            category = REPRESENTATIVE_MODELS[k]["category"]
        else:
            display = OTHER_MODELS[k]["display"]
            category = OTHER_MODELS[k]["category"]

        row, mon, tk = summarize_model(preds[k], display, category)
        summary_rows.append(row)
        monthly_list.append(mon)
        topk_list.append(tk)

    summary = pd.DataFrame(summary_rows)
    monthly = pd.concat(monthly_list, ignore_index=True)
    topk = pd.concat(topk_list, ignore_index=True)

    if "ZeroGraph" in summary["Model"].values:
        zero_r2 = float(summary.loc[summary["Model"] == "ZeroGraph", "OOS_R2_all"].iloc[0])
        summary["Delta_R2_vs_ZeroGraph"] = summary["OOS_R2_all"] - zero_r2
        summary["Pct_R2_vs_ZeroGraph"] = np.where(abs(zero_r2) > 1e-12, summary["Delta_R2_vs_ZeroGraph"] / abs(zero_r2), np.nan)

    non_proposed = summary[summary["Model"] != "Proposed Graph-2"].copy()
    if len(non_proposed):
        best_idx = non_proposed["OOS_R2_all"].idxmax()
        best_model = str(non_proposed.loc[best_idx, "Model"])
        best_r2 = float(non_proposed.loc[best_idx, "OOS_R2_all"])
        summary["Delta_R2_vs_best_baseline"] = summary["OOS_R2_all"] - best_r2
        summary["Best_baseline_reference"] = best_model

    base_cols = [
        "Model", "Category", "N_obs", "N_days",
        "OOS_R2_all", "Weighted_monthly_R2", "MSE", "MAE", "RMSE",
        "IC", "IC_SE", "RankIC", "RankIC_SE",
    ]
    top_cols = []
    for k in TOPK_LIST:
        top_cols.extend([f"Top{k}_mean_y", f"Top{k}_win_rate"])
    extra_cols = [c for c in summary.columns if c not in base_cols + top_cols]
    summary = summary[base_cols + top_cols + extra_cols]

    summary_path = tables_dir / "Table3_SOTA.csv"
    monthly_path = tables_dir / "Table3_SOTA_monthly.csv"
    topk_path = tables_dir / "Table3_SOTA_topk.csv"
    xlsx_path = tables_dir / "Table3_SOTA.xlsx"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    topk.to_csv(topk_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="SOTA_summary")
        monthly.to_excel(writer, index=False, sheet_name="Monthly")
        topk.to_excel(writer, index=False, sheet_name="TopK")
        diag.to_excel(writer, index=False, sheet_name="Input_diagnostics")

        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                max_len = 10
                col_letter = col[0].column_letter
                for cell in col[:200]:
                    max_len = max(max_len, len(str(cell.value)) if cell.value is not None else 0)
                ws.column_dimensions[col_letter].width = min(max_len + 2, 35)

    manifest = {
        "SOTA_RESULTS_REP": SOTA_RESULTS_REP,
        "GCN_LSTM_RESULTS": GCN_LSTM_RESULTS,
        "ZEROGRAPH_TEST_DETAIL": ZEROGRAPH_TEST_DETAIL,
        "GRAPH2_TEST_DETAIL": GRAPH2_TEST_DETAIL,
        "outputs": {
            "summary_csv": str(summary_path),
            "monthly_csv": str(monthly_path),
            "topk_csv": str(topk_path),
            "excel": str(xlsx_path),
            "diagnostics": str(tables_dir / "Table3_SOTA_input_diagnostics.csv"),
        },
        "metric_definitions": {
            "OOS_R2_all": "1 - SSE/SST, pooled over all test stock-day observations.",
            "Weighted_monthly_R2": "Observation-weighted average of month-level R2.",
            "IC": "Mean daily Pearson correlation between y_true and y_pred.",
            "RankIC": "Mean daily Spearman rank correlation between y_true and y_pred.",
            "TopK_mean_y": "Mean daily realized y_true among top-K stocks ranked by y_pred.",
        },
    }
    with open(tables_dir / "Table3_SOTA_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nFinal SOTA summary:")
    print(summary[["Model", "Category", "OOS_R2_all", "Weighted_monthly_R2", "MSE", "MAE", "RMSE", "IC", "RankIC"]].to_string(index=False))

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {monthly_path}")
    print(f"  {topk_path}")
    print(f"  {xlsx_path}")


if __name__ == "__main__":
    main()
