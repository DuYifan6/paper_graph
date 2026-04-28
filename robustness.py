# -*- coding: utf-8 -*-
"""
Robustness driver for single-factor W / k / tau experiments
(weighted summary version with BOTH RealGraph and ZeroGraph reported).

What it does
- Reads a BASE_SCRIPT (e.g. graph2_C_k3510.py / graph1_C_k3510.py)
- Creates patched copies that vary only one factor at a time:
    W   in [10, 20, 40]
    k   in [20, 40, 60]
    tau in [0.05, 0.10, 0.15]
- Runs each script sequentially
- Collects:
    * summary_{exp_name}.csv
    * compare_{exp_name}_vs_Zero_train_by_quarter.csv
    * compare_{exp_name}_vs_Zero_test_by_month.csv
- Builds a final robustness summary table

Key upgrade vs the old version
- Reports BOTH ZeroGraph and RealGraph weighted-average quarterly/monthly R²
- Reports weighted-average delta R²
- Uses n_obs as weights, so the reporting logic matches weighted reporting in the paper
"""

import os
import re
import sys
import subprocess
from typing import Dict, List

import numpy as np
import pandas as pd

# =========================================================
# 1) USER SETTINGS
# =========================================================
# Change this to your base script.
BASE_SCRIPT = r"D:\PythonProject\LASSO_FINAL\graph_调整\graph2_C_k3510.py"

# Root output folder for all robustness runs.
ROBUST_ROOT = r"D:\PythonProject\LASSO_FINAL\robustness\robustness_graph2_C_k3510_zero_60"

# Whether to actually launch training runs.
# Set False if all run folders already exist and you only want to aggregate.
RUN_TRAIN = True

# Whether to overwrite generated patched scripts if they exist.
OVERWRITE_PATCHED_SCRIPT = True

# Python executable
PYTHON_EXE = sys.executable

# Experiment-name prefix used by the base script family
# Examples:
#   Graph1_C_K3510
#   Graph2_C_K3510
#   Graph3_C_K3510
EXP_PREFIX = "Graph2_C_K3510"

# Single-factor robustness design
W_LIST = [10, 20, 40]
K_LIST = [40, 60, 80]
TAU_LIST = [0.05, 0.10, 0.15]

# If you already have the baseline main result and do NOT want to rerun
# the duplicated baseline values (W=20, k=40, tau=0.10) three times,
# set this to True. Then the script keeps only the six off-baseline runs.
ONLY_OFF_BASELINE = True

# =========================================================
# 2) HELPERS
# =========================================================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def replace_one(text: str, pattern: str, replacement: str) -> str:
    new_text, n = re.subn(pattern, lambda m: replacement, text, count=1, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"Pattern not found or not unique: {pattern}")
    return new_text


def patch_script_text(base_text: str, exp_name: str, out_dir: str,
                      roll_w: int, topk: int, edge_tau: float) -> str:
    txt = base_text

    # Patch experiment name
    txt = replace_one(
        txt,
        r'^EXP_NAME\s*=\s*f?"[^"]*"\s*$',
        f'EXP_NAME = "{exp_name}"'
    )

    # Patch robustness parameters
    txt = replace_one(txt, r'^ROLL_W\s*=\s*.*$', f'ROLL_W = {roll_w}')
    txt = replace_one(txt, r'^TOPK\s*=\s*.*$', f'TOPK = {topk}')
    txt = replace_one(txt, r'^EDGE_TAU\s*=\s*.*$', f'EDGE_TAU = {edge_tau:.2f}')

    # Patch OUT_DIR
    txt = replace_one(
        txt,
        r'^OUT_DIR\s*=\s*rf?"[^"]*"\s*$',
        f'OUT_DIR = r"{out_dir}"'
    )

    return txt


def build_setting_specs() -> List[Dict]:
    specs = []
    baseline_w = 20
    baseline_k = 60
    baseline_tau = 0.10

    for w in W_LIST:
        specs.append({
            "family": "W",
            "value": w,
            "roll_w": w,
            "topk": baseline_k,
            "edge_tau": baseline_tau,
            "tag": f"W{w}",
            "is_baseline": (w == baseline_w),
        })

    for k in K_LIST:
        specs.append({
            "family": "k",
            "value": k,
            "roll_w": baseline_w,
            "topk": k,
            "edge_tau": baseline_tau,
            "tag": f"k{k}",
            "is_baseline": (k == baseline_k),
        })

    for tau in TAU_LIST:
        specs.append({
            "family": "tau",
            "value": tau,
            "roll_w": baseline_w,
            "topk": baseline_k,
            "edge_tau": tau,
            "tag": f"tau{tau:.2f}",
            "is_baseline": abs(tau - baseline_tau) < 1e-12,
        })

    if ONLY_OFF_BASELINE:
        specs = [x for x in specs if not x["is_baseline"]]

    return specs


def run_one_script(script_path: str) -> int:
    print(f"\n[RUN] {script_path}")
    proc = subprocess.run([PYTHON_EXE, script_path])
    return int(proc.returncode)


def safe_read_csv(path: str) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise last_err


def pick_first_existing(paths: List[str]) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these files exist: {paths}")


def weighted_mean(series: pd.Series, weight: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weight, errors="coerce")
    mask = s.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float((s[mask] * w[mask]).sum() / w[mask].sum())


def weighted_positive_ratio(delta_series: pd.Series, weight: pd.Series) -> float:
    d = pd.to_numeric(delta_series, errors="coerce")
    w = pd.to_numeric(weight, errors="coerce")
    mask = d.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan
    wsum = float(w[mask].sum())
    if wsum <= 0:
        return np.nan
    return float(w[mask & (d > 0)].sum() / wsum)


def aggregate_one_run(run_dir: str, exp_name: str, family: str, value, is_baseline: bool) -> Dict:
    result_dir = os.path.join(run_dir, "results")

    summary_fp = pick_first_existing([
        os.path.join(result_dir, f"summary_{exp_name}.csv"),
    ])
    test_cmp_fp = pick_first_existing([
        os.path.join(result_dir, f"compare_{exp_name}_vs_Zero_test_by_month.csv"),
    ])
    train_cmp_fp = pick_first_existing([
        os.path.join(result_dir, f"compare_{exp_name}_vs_Zero_train_by_quarter.csv"),
    ])

    df_sum = safe_read_csv(summary_fp)
    df_test = safe_read_csv(test_cmp_fp)
    df_train = safe_read_csv(train_cmp_fp)

    row = df_sum.iloc[0].to_dict()

    # -------------------------
    # weighted train-quarter stats
    # -------------------------
    w_train_col = "n_obs" if "n_obs" in df_train.columns else None
    weighted_r2_zero_train = weighted_mean(df_train["r2_zero"], df_train[w_train_col]) if (w_train_col and "r2_zero" in df_train.columns) else np.nan
    weighted_r2_real_train = weighted_mean(df_train["r2_real"], df_train[w_train_col]) if (w_train_col and "r2_real" in df_train.columns) else np.nan
    weighted_delta_r2_train = weighted_mean(df_train["delta_r2(real-zero)"], df_train[w_train_col]) if (w_train_col and "delta_r2(real-zero)" in df_train.columns) else np.nan
    weighted_train_dm = weighted_mean(df_train["dm_stat"], df_train[w_train_col]) if (w_train_col and "dm_stat" in df_train.columns) else np.nan
    weighted_train_positive_ratio = weighted_positive_ratio(df_train["delta_r2(real-zero)"], df_train[w_train_col]) if (w_train_col and "delta_r2(real-zero)" in df_train.columns) else np.nan

    # -------------------------
    # weighted test-month stats
    # -------------------------
    w_test_col = "n_obs" if "n_obs" in df_test.columns else None
    weighted_r2_zero_test = weighted_mean(df_test["r2_zero"], df_test[w_test_col]) if (w_test_col and "r2_zero" in df_test.columns) else np.nan
    weighted_r2_real_test = weighted_mean(df_test["r2_real"], df_test[w_test_col]) if (w_test_col and "r2_real" in df_test.columns) else np.nan
    weighted_delta_r2_test = weighted_mean(df_test["delta_r2(real-zero)"], df_test[w_test_col]) if (w_test_col and "delta_r2(real-zero)" in df_test.columns) else np.nan
    weighted_test_dm = weighted_mean(df_test["dm_stat"], df_test[w_test_col]) if (w_test_col and "dm_stat" in df_test.columns) else np.nan
    weighted_test_positive_ratio = weighted_positive_ratio(df_test["delta_r2(real-zero)"], df_test[w_test_col]) if (w_test_col and "delta_r2(real-zero)" in df_test.columns) else np.nan

    sig_test_10 = int((pd.to_numeric(df_test["dm_p_one"], errors="coerce") < 0.10).sum()) if "dm_p_one" in df_test.columns else np.nan
    sig_test_05 = int((pd.to_numeric(df_test["dm_p_one"], errors="coerce") < 0.05).sum()) if "dm_p_one" in df_test.columns else np.nan

    out = {
        "family": family,
        "value": value,
        "exp_name": exp_name,
        "is_baseline": int(is_baseline),

        # overall summary metrics from summary_{exp_name}.csv
        "overall_r2_train": row.get("r2_train", np.nan),
        "overall_r2_test": row.get("r2_test", np.nan),
        "best_epoch": row.get("best_epoch", np.nan),
        "best_val_score": row.get("best_val_score", np.nan),

        # weighted quarterly metrics
        "weighted_r2_zero_train_quarter": weighted_r2_zero_train,
        "weighted_r2_real_train_quarter": weighted_r2_real_train,
        "weighted_delta_r2_train_quarter": weighted_delta_r2_train,
        "weighted_train_positive_ratio": weighted_train_positive_ratio,
        "weighted_avg_train_dm_stat": weighted_train_dm,

        # weighted monthly metrics
        "weighted_r2_zero_test_month": weighted_r2_zero_test,
        "weighted_r2_real_test_month": weighted_r2_real_test,
        "weighted_delta_r2_test_month": weighted_delta_r2_test,
        "weighted_test_positive_ratio": weighted_test_positive_ratio,
        "weighted_avg_test_dm_stat": weighted_test_dm,
        "test_dm_sig_10pct_count": sig_test_10,
        "test_dm_sig_5pct_count": sig_test_05,

        # traceability
        "summary_fp": summary_fp,
        "train_compare_fp": train_cmp_fp,
        "test_compare_fp": test_cmp_fp,
    }
    return out


# =========================================================
# 3) MAIN
# =========================================================
def main():
    ensure_dir(ROBUST_ROOT)
    patched_dir = os.path.join(ROBUST_ROOT, "patched_scripts")
    ensure_dir(patched_dir)

    with open(BASE_SCRIPT, "r", encoding="utf-8") as f:
        base_text = f.read()

    specs = build_setting_specs()
    run_rows = []

    for spec in specs:
        exp_name = f"{EXP_PREFIX}_robust_{spec['tag']}"
        run_dir = os.path.join(ROBUST_ROOT, exp_name)
        ensure_dir(run_dir)

        patched_script = os.path.join(patched_dir, f"{exp_name}.py")
        patched_text = patch_script_text(
            base_text=base_text,
            exp_name=exp_name,
            out_dir=run_dir,
            roll_w=spec["roll_w"],
            topk=spec["topk"],
            edge_tau=spec["edge_tau"],
        )

        if OVERWRITE_PATCHED_SCRIPT or (not os.path.exists(patched_script)):
            with open(patched_script, "w", encoding="utf-8") as f:
                f.write(patched_text)

        if RUN_TRAIN:
            code = run_one_script(patched_script)
            if code != 0:
                raise RuntimeError(f"Run failed with exit code {code}: {patched_script}")

        row = aggregate_one_run(
            run_dir=run_dir,
            exp_name=exp_name,
            family=spec["family"],
            value=spec["value"],
            is_baseline=spec["is_baseline"],
        )
        run_rows.append(row)

    df = pd.DataFrame(run_rows)

    family_order = {"W": 0, "k": 1, "tau": 2}
    df["family_order"] = df["family"].map(family_order)
    df = df.sort_values(["family_order", "value"]).drop(columns=["family_order"]).reset_index(drop=True)

    out_csv = os.path.join(ROBUST_ROOT, "robustness_summary_W_k_tau.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    for fam in ["W", "k", "tau"]:
        sub = df[df["family"] == fam].copy().reset_index(drop=True)
        sub.to_csv(os.path.join(ROBUST_ROOT, f"robustness_summary_{fam}.csv"), index=False, encoding="utf-8-sig")

    print("\n===== Robustness summary saved =====")
    print(out_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
