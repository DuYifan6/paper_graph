#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 3: Plot ESWA SOTA figures.

Generate:
    Figure6_Monthly_R2
    Figure7_Monthly_RankIC
    Figure8_Top1_CumulativeReturn
    Figure9_Top5_CumulativeReturn
    Figure10_Summary_OOS_R2

Inputs:
    final_eswa_sota/tables/Table3_SOTA.csv
    final_eswa_sota/tables/Table3_SOTA_monthly.csv
    prediction-detail files for return curves

Run:
D:\Anaconda3\python.exe D:\PythonProject\LASSO_FINAL\SOTA\plot_sota_figures.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 0) PATH CONFIGURATION
# =============================================================================

SOTA_RESULTS_REP = r"D:\PythonProject\LASSO_FINAL\SOTA\sota_results_rep"
GCN_LSTM_RESULTS = r"D:\PythonProject\LASSO_FINAL\SOTA\sota_results_gcn_lstm"

ZEROGRAPH_TEST_DETAIL = r"D:\PythonProject\LASSO_FINAL\graph_调整\fixed_zerograph_valselect\results\predictions_ZeroGraph_test_detail.csv"
GRAPH2_TEST_DETAIL = r"D:\PythonProject\LASSO_FINAL\graph_调整\graph2_binary_60\results\predictions_Graph2_C_K3510_test_detail.csv"

FINAL_DIR = r"D:\PythonProject\LASSO_FINAL\SOTA\final_eswa_sota"
TABLE_DIR = os.path.join(FINAL_DIR, "tables")
FIG_DIR = os.path.join(FINAL_DIR, "figures")

TABLE3 = os.path.join(TABLE_DIR, "Table3_SOTA.csv")
MONTHLY = os.path.join(TABLE_DIR, "Table3_SOTA_monthly.csv")

DPI = 300

# Journal-friendly fixed colors. You can change these if needed.
MODEL_COLORS = {
    "Proposed Graph-2": "#1f77b4",
    "ZeroGraph": "#ff7f0e",
    "GCN-LSTM": "#2ca02c",
    "LightGBM": "#d62728",
    "Transformer": "#9467bd",
    "Ridge": "#8c564b",
    "MLP": "#7f7f7f",
}

MODEL_ORDER = [
    "Proposed Graph-2",
    "ZeroGraph",
    "GCN-LSTM",
    "LightGBM",
    "Transformer",
    "Ridge",
    "MLP",
]

RETURN_MODELS = [
    "Proposed Graph-2",
    "ZeroGraph",
    "LightGBM",
    "GCN-LSTM",
]


PRED_PATHS = {
    "Ridge": os.path.join(SOTA_RESULTS_REP, "pred_ridge.parquet"),
    "LightGBM": os.path.join(SOTA_RESULTS_REP, "pred_lightgbm.parquet"),
    "MLP": os.path.join(SOTA_RESULTS_REP, "pred_mlp.parquet"),
    "Transformer": os.path.join(SOTA_RESULTS_REP, "pred_transformer.parquet"),
    "GCN-LSTM": os.path.join(GCN_LSTM_RESULTS, "pred_gcn_lstm.csv"),
    "ZeroGraph": ZEROGRAPH_TEST_DETAIL,
    "Proposed Graph-2": GRAPH2_TEST_DETAIL,
}


# =============================================================================
# 1) GENERAL HELPERS
# =============================================================================

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


def save_figure(fig, name: str):
    fig_dir = ensure_dir(FIG_DIR)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(fig_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / (name + '.pdf')}")


def read_pred(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in [".csv", ".txt"]:
        return pd.read_csv(path)
    if path.suffix.lower() in [".pkl", ".pickle"]:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported file: {path}")


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def standardize_pred_df(raw: pd.DataFrame, model: str) -> pd.DataFrame:
    df = raw.copy()
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed:")]]

    date_col = find_col(df, ["date", "trade_date", "datetime", "cal_date", "day"])
    stock_col = find_col(df, ["stock", "ts_code", "code", "ticker", "symbol", "n_idx", "node", "node_id"])
    y_true_col = find_col(df, ["y_true", "true", "actual", "real", "label", "target", "y", "gap_up_pct", "overnight_return"])
    y_pred_col = find_col(df, ["y_pred", "pred", "prediction", "forecast", "yhat", "y_hat", "score"])

    if y_pred_col is None:
        pred_like = [c for c in df.columns if "pred" in str(c).lower()]
        if len(pred_like) == 1:
            y_pred_col = pred_like[0]
    if y_true_col is None:
        true_like = [c for c in df.columns if any(x in str(c).lower() for x in ["true", "actual", "real", "target", "label"])]
        if len(true_like) == 1:
            y_true_col = true_like[0]

    missing = []
    for name, col in [("date", date_col), ("stock", stock_col), ("y_true", y_true_col), ("y_pred", y_pred_col)]:
        if col is None:
            missing.append(name)
    if missing:
        raise KeyError(f"{model}: missing {missing}; columns={list(df.columns)}")

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d"),
        "stock": df[stock_col].astype(str),
        "y_true": pd.to_numeric(df[y_true_col], errors="coerce"),
        "y_pred": pd.to_numeric(df[y_pred_col], errors="coerce"),
        "model": model,
    })

    if "n_idx" in df.columns:
        out["n_idx"] = pd.to_numeric(df["n_idx"], errors="coerce")
    else:
        numeric_stock = pd.to_numeric(out["stock"], errors="coerce")
        if numeric_stock.notna().mean() > 0.95:
            out["n_idx"] = numeric_stock
        else:
            out["n_idx"] = out.groupby("date", sort=False).cumcount()

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "n_idx", "y_true", "y_pred"])
    out["n_idx"] = out["n_idx"].astype(int)
    out = out.drop_duplicates(["date", "n_idx"], keep="last")
    return out


def load_model_pred(model: str) -> pd.DataFrame:
    return standardize_pred_df(read_pred(PRED_PATHS[model]), model)


# =============================================================================
# 2) MONTHLY METRIC FIGURES
# =============================================================================

def plot_monthly_metric(metric: str, title: str, ylabel: str, out_name: str):
    monthly = pd.read_csv(MONTHLY)
    monthly["month"] = monthly["month"].astype(str)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    for model in MODEL_ORDER:
        if model not in monthly["model"].unique():
            continue
        g = monthly[monthly["model"] == model].sort_values("month")
        ax.plot(
            g["month"],
            g[metric],
            marker="o",
            linewidth=1.8 if model in ["Proposed Graph-2", "ZeroGraph"] else 1.3,
            markersize=4,
            label=model,
            color=MODEL_COLORS.get(model, None),
            alpha=1.0 if model in ["Proposed Graph-2", "ZeroGraph"] else 0.85,
        )

    ax.axhline(0, linewidth=0.8, color="black", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(ncol=2, frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, out_name)


def figure6_monthly_r2():
    plot_monthly_metric(
        metric="r2",
        title="Monthly out-of-sample R² across representative models",
        ylabel="OOS R²",
        out_name="Figure6_Monthly_R2",
    )


def figure7_monthly_rankic():
    plot_monthly_metric(
        metric="rank_ic",
        title="Monthly RankIC across representative models",
        ylabel="RankIC",
        out_name="Figure7_Monthly_RankIC",
    )


# =============================================================================
# 3) CUMULATIVE RETURN FIGURES
# =============================================================================

def topk_daily_return(df: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = []
    for d, g in df.groupby("date", sort=True):
        top = g.sort_values("y_pred", ascending=False).head(k)
        if len(top) == 0:
            continue
        # y_true is percentage return in your pipeline.
        r = float(top["y_true"].mean()) / 100.0
        rows.append({"date": d, "ret": r})
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    out["equity"] = (1.0 + out["ret"]).cumprod()
    return out


def plot_cumulative_return(k: int, out_name: str, title: str):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    for model in RETURN_MODELS:
        df = load_model_pred(model)
        curve = topk_daily_return(df, k=k)
        if len(curve) == 0:
            continue
        ax.plot(
            curve["date"],
            curve["equity"],
            label=model,
            linewidth=2.0 if model == "Proposed Graph-2" else 1.6,
            color=MODEL_COLORS.get(model, None),
        )

    ax.axhline(1.0, linewidth=0.8, color="black", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative gross return")
    ax.legend(frameon=False, loc="best")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    save_figure(fig, out_name)


def figure8_top1_return():
    plot_cumulative_return(
        k=1,
        out_name="Figure8_Top1_CumulativeReturn",
        title="Top-1 cumulative gross return on the holdout sample",
    )


def figure9_top5_return():
    plot_cumulative_return(
        k=5,
        out_name="Figure9_Top5_CumulativeReturn",
        title="Top-5 cumulative gross return on the holdout sample",
    )


# =============================================================================
# 4) SUMMARY BAR FIGURE
# =============================================================================

def figure10_summary_bar():
    table = pd.read_csv(TABLE3)
    table["Model"] = pd.Categorical(table["Model"], categories=MODEL_ORDER[::-1], ordered=True)
    table = table.sort_values("Model")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = [MODEL_COLORS.get(str(m), None) for m in table["Model"]]

    ax.barh(table["Model"].astype(str), table["OOS_R2_all"], color=colors, alpha=0.9)
    ax.axvline(0, linewidth=0.8, color="black", alpha=0.6)
    ax.set_title("Overall out-of-sample R² by model")
    ax.set_xlabel("OOS R²")
    ax.set_ylabel("")
    for i, v in enumerate(table["OOS_R2_all"]):
        ax.text(v + (0.002 if v >= 0 else -0.002), i, f"{v:.4f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "Figure10_Summary_OOS_R2")


# =============================================================================
# 5) MAIN
# =============================================================================

def main():
    ensure_dir(FIG_DIR)
    set_plot_style()

    print("Generating ESWA SOTA figures...")
    figure6_monthly_r2()
    figure7_monthly_rankic()
    figure8_top1_return()
    figure9_top5_return()
    figure10_summary_bar()
    print(f"\nDone. Figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
