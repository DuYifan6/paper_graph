#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2: Run DM tests for ESWA SOTA benchmark.

Computes daily cross-sectional MSE loss differentials:
    d_t = MSE_baseline(t) - MSE_candidate(t)

H1: mean(d_t) > 0, candidate is better than baseline.

Run:
D:\Anaconda3\python.exe D:\PythonProject\LASSO_FINAL\SOTA\run_dm_tests.py
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


SOTA_RESULTS_REP = r"D:\PythonProject\LASSO_FINAL\SOTA\sota_results_rep"
GCN_LSTM_RESULTS = r"D:\PythonProject\LASSO_FINAL\SOTA\sota_results_gcn_lstm"

ZEROGRAPH_TEST_DETAIL = r"D:\PythonProject\LASSO_FINAL\graph_调整\fixed_zerograph_valselect\results\predictions_ZeroGraph_test_detail.csv"
GRAPH2_TEST_DETAIL = r"D:\PythonProject\LASSO_FINAL\graph_调整\graph2_binary_60\results\predictions_Graph2_C_K3510_test_detail.csv"

OUT_DIR = r"D:\PythonProject\LASSO_FINAL\SOTA\final_eswa_sota"
DM_LAG = 2

REPRESENTATIVE_MODELS = {
    "ridge": {"display": "Ridge", "file_stem": "pred_ridge"},
    "lightgbm": {"display": "LightGBM", "file_stem": "pred_lightgbm"},
    "mlp": {"display": "MLP", "file_stem": "pred_mlp"},
    "transformer": {"display": "Transformer", "file_stem": "pred_transformer"},
}

OTHER_MODELS = {
    "gcn_lstm": {"display": "GCN-LSTM", "path": os.path.join(GCN_LSTM_RESULTS, "pred_gcn_lstm.csv")},
    "zerograph": {"display": "ZeroGraph", "path": ZEROGRAPH_TEST_DETAIL},
    "graph2": {"display": "Proposed Graph-2", "path": GRAPH2_TEST_DETAIL},
}

COMPARISONS = [
    ("graph2", "zerograph"),
    ("graph2", "lightgbm"),
    ("graph2", "gcn_lstm"),
    ("graph2", "transformer"),
    ("graph2", "ridge"),
    ("graph2", "mlp"),
]


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

    # Keep or reconstruct n_idx for cross-model alignment.
    # SOTA baseline files often store stock as node id 0,1,2,...
    # Graph2/ZeroGraph files often store stock as real ts_code, e.g. 000001.SZ.
    if "n_idx" in df.columns:
        out["n_idx"] = pd.to_numeric(df["n_idx"], errors="coerce")
    else:
        numeric_stock = pd.to_numeric(out["stock"], errors="coerce")
        if numeric_stock.notna().mean() > 0.95:
            out["n_idx"] = numeric_stock
        else:
            # For Graph2/ZeroGraph: create node index by within-date row order.
            # This is valid because their detail files are saved in the same tensor stock order.
            out["n_idx"] = out.groupby("date", sort=False).cumcount()

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "n_idx", "y_true", "y_pred"])
    out["n_idx"] = out["n_idx"].astype(int)

    # Deduplicate by date-node rather than date-stock, because model files use different stock labels.
    out = out.drop_duplicates(["date", "n_idx"], keep="last")
    return out


def load_rep_model_predictions(model_key: str, cfg: Dict) -> pd.DataFrame:
    root = Path(SOTA_RESULTS_REP)
    stem = cfg["file_stem"]
    possible = [root / f"{stem}.parquet", root / f"{stem}.csv", root / f"{stem}.pkl"]
    path = find_first_existing(possible)
    if path is None:
        raise FileNotFoundError("Cannot find prediction file for " + model_key + ". Tried:\n" + "\n".join(str(p) for p in possible))
    df = standardize_prediction_df(read_prediction_file(path), cfg["display"], path)
    print(f"[OK] {cfg['display']}: {len(df):,} rows from {path}")
    return df


def load_other_model_predictions(model_key: str, cfg: Dict) -> pd.DataFrame:
    path = Path(cfg["path"])
    df = standardize_prediction_df(read_prediction_file(path), cfg["display"], path)
    print(f"[OK] {cfg['display']}: {len(df):,} rows from {path}")
    return df


def load_all_predictions() -> Dict[str, pd.DataFrame]:
    out = {}
    for k, cfg in REPRESENTATIVE_MODELS.items():
        out[k] = load_rep_model_predictions(k, cfg)
    for k, cfg in OTHER_MODELS.items():
        out[k] = load_other_model_predictions(k, cfg)
    return out


def display_name(model_key: str) -> str:
    if model_key in REPRESENTATIVE_MODELS:
        return REPRESENTATIVE_MODELS[model_key]["display"]
    return OTHER_MODELS[model_key]["display"]


def newey_west_se(x: np.ndarray, lag: int = 2) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T <= lag + 1:
        return np.nan
    xc = x - x.mean()
    gamma0 = np.sum(xc * xc) / T
    var = gamma0
    for l in range(1, lag + 1):
        gamma = np.sum(xc[l:] * xc[:-l]) / T
        var += 2.0 * (1.0 - l / (lag + 1.0)) * gamma
    return math.sqrt(max(var, 0.0) / T)


def paired_daily_loss_differential(candidate: pd.DataFrame, baseline: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Align by date + n_idx, not by stock label.
    # This fixes the mismatch where SOTA baselines use node ids while Graph2/ZeroGraph use real ts_codes.
    cand = candidate[["date", "n_idx", "stock", "y_true", "y_pred"]].copy()
    base = baseline[["date", "n_idx", "stock", "y_pred"]].copy()

    cand["date"] = cand["date"].astype(str)
    base["date"] = base["date"].astype(str)
    cand["n_idx"] = cand["n_idx"].astype(int)
    base["n_idx"] = base["n_idx"].astype(int)

    merged = cand.merge(base, on=["date", "n_idx"], how="inner", suffixes=("_candidate", "_baseline"))
    if len(merged) == 0:
        return pd.DataFrame(), merged

    merged["err_candidate"] = (merged["y_true"] - merged["y_pred_candidate"]) ** 2
    merged["err_baseline"] = (merged["y_true"] - merged["y_pred_baseline"]) ** 2
    merged["loss_diff"] = merged["err_baseline"] - merged["err_candidate"]

    daily = merged.groupby("date", sort=True).agg(
        n_obs=("loss_diff", "size"),
        mse_candidate=("err_candidate", "mean"),
        mse_baseline=("err_baseline", "mean"),
        loss_diff=("loss_diff", "mean"),
    ).reset_index()
    return daily, merged


def dm_test(candidate: pd.DataFrame, baseline: pd.DataFrame, candidate_name: str, baseline_name: str, lag: int = 2):
    daily, merged = paired_daily_loss_differential(candidate, baseline)

    if len(daily) == 0:
        row = {
            "Candidate": candidate_name, "Baseline": baseline_name,
            "Comparison": f"{candidate_name} vs {baseline_name}",
            "N_days": 0, "N_obs_common": 0,
            "Mean_loss_diff": np.nan, "NW_SE": np.nan,
            "DM_stat": np.nan, "p_one_sided_greater": np.nan, "p_two_sided": np.nan,
            "Mean_MSE_candidate": np.nan, "Mean_MSE_baseline": np.nan,
            "Positive_day_ratio": np.nan, "Interpretation": "No common observations",
        }
        daily["Candidate"] = candidate_name
        daily["Baseline"] = baseline_name
        return row, daily

    d = daily["loss_diff"].to_numpy(dtype=float)
    mean_d = float(np.mean(d))
    se = newey_west_se(d, lag=lag)
    dm = mean_d / se if np.isfinite(se) and se > 0 else np.nan
    p_one = float(1.0 - norm.cdf(dm)) if np.isfinite(dm) else np.nan
    p_two = float(2.0 * min(norm.cdf(dm), 1.0 - norm.cdf(dm))) if np.isfinite(dm) else np.nan

    interp = "candidate better" if mean_d > 0 else "baseline better"
    if np.isfinite(p_one):
        if p_one < 0.01:
            interp += " at 1%"
        elif p_one < 0.05:
            interp += " at 5%"
        elif p_one < 0.10:
            interp += " at 10%"

    row = {
        "Candidate": candidate_name,
        "Baseline": baseline_name,
        "Comparison": f"{candidate_name} vs {baseline_name}",
        "N_days": int(len(daily)),
        "N_obs_common": int(len(merged)),
        "Mean_loss_diff": mean_d,
        "NW_SE": float(se) if np.isfinite(se) else np.nan,
        "DM_stat": float(dm) if np.isfinite(dm) else np.nan,
        "p_one_sided_greater": p_one,
        "p_two_sided": p_two,
        "Mean_MSE_candidate": float(daily["mse_candidate"].mean()),
        "Mean_MSE_baseline": float(daily["mse_baseline"].mean()),
        "Positive_day_ratio": float((daily["loss_diff"] > 0).mean()),
        "Interpretation": interp,
    }

    daily = daily.copy()
    daily["Candidate"] = candidate_name
    daily["Baseline"] = baseline_name
    daily["Comparison"] = f"{candidate_name} vs {baseline_name}"
    return row, daily


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    tables_dir = ensure_dir(Path(OUT_DIR) / "tables")

    print("=" * 90)
    print("Running DM tests for ESWA SOTA benchmark")
    print("=" * 90)

    preds = load_all_predictions()
    rows, daily_all = [], []

    for cand_key, base_key in COMPARISONS:
        cand_name = display_name(cand_key)
        base_name = display_name(base_key)
        print(f"\nDM: {cand_name} vs {base_name}")

        row, daily = dm_test(preds[cand_key], preds[base_key], cand_name, base_name, lag=DM_LAG)
        rows.append(row)
        daily_all.append(daily)

        print(
            f"  DM={row['DM_stat']:.4f}, p(one-sided)={row['p_one_sided_greater']:.4g}, "
            f"mean_d={row['Mean_loss_diff']:.6g}, days={row['N_days']}, common_obs={row['N_obs_common']}"
        )

    dm_table = pd.DataFrame(rows)
    daily_df = pd.concat(daily_all, ignore_index=True) if daily_all else pd.DataFrame()

    out_csv = tables_dir / "Table4_DM.csv"
    out_daily = tables_dir / "Table4_DM_daily_loss_differentials.csv"
    out_xlsx = tables_dir / "Table4_DM.xlsx"

    dm_table.to_csv(out_csv, index=False, encoding="utf-8-sig")
    daily_df.to_csv(out_daily, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        dm_table.to_excel(writer, index=False, sheet_name="DM_summary")
        daily_df.to_excel(writer, index=False, sheet_name="Daily_loss_diff")
        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                col_letter = col[0].column_letter
                max_len = 10
                for cell in col[:200]:
                    max_len = max(max_len, len(str(cell.value)) if cell.value is not None else 0)
                ws.column_dimensions[col_letter].width = min(max_len + 2, 35)

    manifest = {
        "DM_LAG": DM_LAG,
        "definition": "d_t = MSE_baseline(t) - MSE_candidate(t); H1: mean(d_t) > 0.",
        "comparisons": [{"candidate": display_name(c), "baseline": display_name(b)} for c, b in COMPARISONS],
        "outputs": {
            "dm_csv": str(out_csv),
            "daily_loss_diff_csv": str(out_daily),
            "excel": str(out_xlsx),
        },
    }
    with open(tables_dir / "Table4_DM_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nFinal DM table:")
    print(dm_table[[
        "Comparison", "N_days", "N_obs_common", "Mean_loss_diff", "DM_stat",
        "p_one_sided_greater", "Positive_day_ratio", "Interpretation"
    ]].to_string(index=False))

    print("\nSaved:")
    print(f"  {out_csv}")
    print(f"  {out_daily}")
    print(f"  {out_xlsx}")


if __name__ == "__main__":
    main()
