# -*- coding: utf-8 -*-
"""
Robustness driver for single-factor W / k / tau experiments.

How it works
- Reads a BASE_SCRIPT (e.g. graph1_C_k3510.py)
- Creates 9 patched copies that vary only one factor at a time:
    W   in [10, 20, 40]
    k   in [20, 40, 60]
    tau in [0.05, 0.10, 0.15]
- Runs each script sequentially
- Collects summary / compare csv outputs
- Builds a final robustness summary table

Designed for the user's Graph1/2/3 C_K3510-style scripts where the code
contains constants such as ROLL_W, TOPK, EDGE_TAU, EXP_NAME, OUT_DIR.
"""

import os
import re
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# =========================================================
# 1) USER SETTINGS
# =========================================================
# Change this to your base script.
BASE_SCRIPT = r"D:\PythonProject\LASSO_FINAL\FINAL-code\graph2.py"

# Root output folder for all robustness runs.
ROBUST_ROOT = r"D:\PythonProject\LASSO_FINAL\robustness\robustness_graph2"

# Whether to actually launch training runs.
# Set False if all run folders already exist and you only want to aggregate.
RUN_TRAIN = True

# Whether to overwrite generated patched scripts if they exist.
OVERWRITE_PATCHED_SCRIPT = True

# The python executable to use.
PYTHON_EXE = sys.executable

# Single-factor robustness design.
W_LIST = [10, 20, 40]
K_LIST = [20, 40, 60]
TAU_LIST = [0.05, 0.10, 0.15]

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

    # Patch core robustness parameters
    txt = replace_one(txt, r'^ROLL_W\s*=\s*.*$', f'ROLL_W = {roll_w}')
    txt = replace_one(txt, r'^TOPK\s*=\s*.*$', f'TOPK = {topk}')
    txt = replace_one(txt, r'^EDGE_TAU\s*=\s*.*$', f'EDGE_TAU = {edge_tau:.2f}')

    # Patch output directory
    txt = replace_one(
        txt,
        r'^OUT_DIR\s*=\s*rf?"[^"]*"\s*$',
        f'OUT_DIR = r"{out_dir}"'
    )

    return txt


def build_setting_specs() -> List[Dict]:
    specs = []
    baseline_w = 20
    baseline_k = 40
    baseline_tau = 0.10

    for w in W_LIST:
        specs.append({
            "family": "W",
            "value": w,
            "roll_w": w,
            "topk": baseline_k,
            "edge_tau": baseline_tau,
            "tag": f"W{w}",
        })

    for k in K_LIST:
        specs.append({
            "family": "k",
            "value": k,
            "roll_w": baseline_w,
            "topk": k,
            "edge_tau": baseline_tau,
            "tag": f"k{k}",
        })

    for tau in TAU_LIST:
        specs.append({
            "family": "tau",
            "value": tau,
            "roll_w": baseline_w,
            "topk": baseline_k,
            "edge_tau": tau,
            "tag": f"tau{tau:.2f}",
        })

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


def aggregate_one_run(run_dir: str, exp_name: str, family: str, value) -> Dict:
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

    # test-month aggregates
    avg_test_delta_r2 = float(df_test["delta_r2(real-zero)"].mean()) if len(df_test) > 0 else np.nan
    pos_test_ratio = float((df_test["delta_r2(real-zero)"] > 0).mean()) if len(df_test) > 0 else np.nan
    avg_test_dm = float(df_test["dm_stat"].mean()) if "dm_stat" in df_test.columns and len(df_test) > 0 else np.nan
    sig_test_10 = int((df_test["dm_p_one"] < 0.10).sum()) if "dm_p_one" in df_test.columns else np.nan
    sig_test_05 = int((df_test["dm_p_one"] < 0.05).sum()) if "dm_p_one" in df_test.columns else np.nan

    # train-quarter aggregates
    avg_train_delta_r2 = float(df_train["delta_r2(real-zero)"].mean()) if len(df_train) > 0 else np.nan
    pos_train_ratio = float((df_train["delta_r2(real-zero)"] > 0).mean()) if len(df_train) > 0 else np.nan
    avg_train_dm = float(df_train["dm_stat"].mean()) if "dm_stat" in df_train.columns and len(df_train) > 0 else np.nan

    out = {
        "family": family,
        "value": value,
        "exp_name": exp_name,
        "r2_train": row.get("r2_train", np.nan),
        "r2_test": row.get("r2_test", np.nan),
        "best_epoch": row.get("best_epoch", np.nan),
        "best_val_score": row.get("best_val_score", np.nan),
        "avg_train_delta_r2": avg_train_delta_r2,
        "train_positive_ratio": pos_train_ratio,
        "avg_train_dm_stat": avg_train_dm,
        "avg_test_delta_r2": avg_test_delta_r2,
        "test_positive_ratio": pos_test_ratio,
        "avg_test_dm_stat": avg_test_dm,
        "test_dm_sig_10pct_count": sig_test_10,
        "test_dm_sig_5pct_count": sig_test_05,
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
        exp_name = f"Graph1_C_K3510_robust_{spec['tag']}"
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
        )
        run_rows.append(row)

    df = pd.DataFrame(run_rows)

    # order rows
    family_order = {"W": 0, "k": 1, "tau": 2}
    df["family_order"] = df["family"].map(family_order)
    df = df.sort_values(["family_order", "value"]).drop(columns=["family_order"]).reset_index(drop=True)

    out_csv = os.path.join(ROBUST_ROOT, "robustness_summary_W_k_tau.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # family split outputs
    for fam in ["W", "k", "tau"]:
        sub = df[df["family"] == fam].copy().reset_index(drop=True)
        sub.to_csv(os.path.join(ROBUST_ROOT, f"robustness_summary_{fam}.csv"), index=False, encoding="utf-8-sig")

    print("\n===== Robustness summary saved =====")
    print(out_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
