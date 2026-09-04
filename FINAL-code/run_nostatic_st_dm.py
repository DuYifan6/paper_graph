#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from project_config import OUTPUT_ROOT

try:
    from scipy.stats import t as student_t

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


DEFAULT_ROOT = OUTPUT_ROOT / "no_static_st"
DEFAULT_ZERO = DEFAULT_ROOT / "zerograph" / "results" / "pred_zerograph_test.csv"
DEFAULT_GRAPH2 = (
    DEFAULT_ROOT
    / "graph2_positive"
    / "results"
    / "predictions_Graph2_Positive_NoStaticST_MSE_test_detail.csv"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "dm_test"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-static-ST Graph-2 vs ZeroGraph DM test.")
    parser.add_argument("--zero-predictions", type=Path, default=DEFAULT_ZERO)
    parser.add_argument("--graph2-predictions", type=Path, default=DEFAULT_GRAPH2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lag", type=int, default=2)
    return parser.parse_args(argv)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb2312"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception as exc:
            last_error = exc
    raise last_error


def normalize_predictions(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    aliases = {"stock": ("stock", "ts_code"), "y_pred": ("y_pred", "prediction"), "y_true": ("y_true", "target")}
    renamed = frame.copy()
    for canonical, candidates in aliases.items():
        if canonical in renamed.columns:
            continue
        found = next((name for name in candidates if name in renamed.columns), None)
        if found is not None:
            renamed = renamed.rename(columns={found: canonical})
    required = ["date", "stock", "y_pred", "y_true"]
    missing = [column for column in required if column not in renamed.columns]
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")

    out = renamed[required].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["stock"] = out["stock"].astype(str).str.strip().str.upper()
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    if out.isna().any().any():
        raise RuntimeError(f"{label} contains missing or invalid required values.")
    duplicate_count = int(out.duplicated(["date", "stock"]).sum())
    if duplicate_count:
        raise RuntimeError(f"{label} contains {duplicate_count} duplicate date-stock rows.")
    return out.sort_values(["date", "stock"]).reset_index(drop=True)


def align_predictions(zero: pd.DataFrame, graph: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    merged = zero.rename(
        columns={"y_pred": "y_pred_zero", "y_true": "y_true_zero"}
    ).merge(
        graph.rename(columns={"y_pred": "y_pred_graph2", "y_true": "y_true_graph2"}),
        on=["date", "stock"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    only_zero = int((merged["_merge"] == "left_only").sum())
    only_graph = int((merged["_merge"] == "right_only").sum())
    both = int((merged["_merge"] == "both").sum())
    audit = {
        "zero_rows": len(zero),
        "graph2_rows": len(graph),
        "matched_rows": both,
        "only_zero": only_zero,
        "only_graph2": only_graph,
        "zero_dates": int(zero["date"].nunique()),
        "graph2_dates": int(graph["date"].nunique()),
        "zero_stocks": int(zero["stock"].nunique()),
        "graph2_stocks": int(graph["stock"].nunique()),
    }
    if only_zero or only_graph:
        raise RuntimeError(
            "Prediction samples differ: "
            f"only_zero={only_zero}, only_graph2={only_graph}, matched={both}."
        )
    merged = merged.drop(columns="_merge")
    max_target_diff = float(
        np.max(np.abs(merged["y_true_zero"] - merged["y_true_graph2"]))
    )
    audit["max_abs_target_difference"] = max_target_diff
    if max_target_diff > 1e-8:
        raise RuntimeError(f"Target mismatch between models: max difference={max_target_diff}.")
    merged["y_true"] = merged.pop("y_true_zero")
    merged = merged.drop(columns="y_true_graph2")
    return merged, audit


def r2_score_np(y_true: np.ndarray, prediction: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = float(np.sum((y_true - prediction) ** 2))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    return float(1.0 - residual / total) if total > 0 else float("nan")


def hac_mean_test(values: np.ndarray, lag: int = 2) -> Dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(len(x))
    if n < 5:
        raise ValueError("At least five daily loss differences are required.")
    lag = int(max(0, min(lag, n - 1)))
    mean = float(x.mean())
    centered = x - mean
    long_run_variance = float(np.dot(centered, centered) / n)
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1.0)
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / n)
        long_run_variance += 2.0 * weight * covariance
    variance_of_mean = max(long_run_variance / n, 1e-12)
    standard_error = math.sqrt(variance_of_mean)
    statistic = mean / standard_error
    if HAS_SCIPY:
        cdf = float(student_t.cdf(statistic, df=n - 1))
        reference = "Student-t"
    else:
        cdf = 0.5 * (1.0 + math.erf(statistic / math.sqrt(2.0)))
        reference = "normal fallback"
    return {
        "T": n,
        "lag": lag,
        "mean_loss_difference_zero_minus_graph2": mean,
        "hac_standard_error": standard_error,
        "dm_statistic": float(statistic),
        "p_one_sided_graph2_better": float(1.0 - cdf),
        "p_two_sided": float(2.0 * min(cdf, 1.0 - cdf)),
        "ci95_low": float(mean - 1.96 * standard_error),
        "ci95_high": float(mean + 1.96 * standard_error),
        "reference_distribution": reference,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    zero = normalize_predictions(safe_read_csv(args.zero_predictions), "ZeroGraph")
    graph = normalize_predictions(safe_read_csv(args.graph2_predictions), "Graph-2")
    merged, audit = align_predictions(zero, graph)
    if merged["date"].min() < pd.Timestamp("2025-01-02") or merged["date"].max() > pd.Timestamp("2025-11-17"):
        raise RuntimeError("Prediction files extend outside the final OOS interval.")

    merged["sqerr_zero"] = (merged["y_true"] - merged["y_pred_zero"]) ** 2
    merged["sqerr_graph2"] = (merged["y_true"] - merged["y_pred_graph2"]) ** 2
    daily = (
        merged.groupby("date", as_index=False)
        .agg(
            nobs=("stock", "size"),
            mse_zero=("sqerr_zero", "mean"),
            mse_graph2=("sqerr_graph2", "mean"),
        )
        .sort_values("date")
    )
    daily["loss_difference_zero_minus_graph2"] = daily["mse_zero"] - daily["mse_graph2"]
    daily["graph2_better"] = daily["loss_difference_zero_minus_graph2"] > 0
    dm = hac_mean_test(daily["loss_difference_zero_minus_graph2"].to_numpy(), args.lag)

    y = merged["y_true"].to_numpy(dtype=float)
    pred_zero = merged["y_pred_zero"].to_numpy(dtype=float)
    pred_graph = merged["y_pred_graph2"].to_numpy(dtype=float)
    mse_zero = float(np.mean((y - pred_zero) ** 2))
    mse_graph = float(np.mean((y - pred_graph) ** 2))
    summary = {
        "n_observations": int(len(merged)),
        "n_days": int(len(daily)),
        "n_stocks": int(merged["stock"].nunique()),
        "pooled_mse_zero": mse_zero,
        "pooled_mse_graph2": mse_graph,
        "mse_reduction_pct": float(100.0 * (mse_zero - mse_graph) / mse_zero),
        "pooled_r2_zero": r2_score_np(y, pred_zero),
        "pooled_r2_graph2": r2_score_np(y, pred_graph),
        "delta_r2_graph2_minus_zero": float(
            r2_score_np(y, pred_graph) - r2_score_np(y, pred_zero)
        ),
        "graph2_better_days": int(daily["graph2_better"].sum()),
        "graph2_better_day_ratio": float(daily["graph2_better"].mean()),
        **dm,
    }

    monthly = daily.assign(month=daily["date"].dt.to_period("M").astype(str)).groupby(
        "month", as_index=False
    ).agg(
        days=("date", "size"),
        mean_mse_zero=("mse_zero", "mean"),
        mean_mse_graph2=("mse_graph2", "mean"),
        mean_loss_difference=("loss_difference_zero_minus_graph2", "mean"),
        graph2_better_day_ratio=("graph2_better", "mean"),
    )

    pd.DataFrame([audit]).to_csv(
        output_dir / "dm_nostatic_st_alignment_audit.csv", index=False, encoding="utf-8-sig"
    )
    daily.to_csv(
        output_dir / "dm_nostatic_st_daily_loss.csv", index=False, encoding="utf-8-sig"
    )
    monthly.to_csv(
        output_dir / "dm_nostatic_st_by_month.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([summary]).to_csv(
        output_dir / "dm_nostatic_st_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
