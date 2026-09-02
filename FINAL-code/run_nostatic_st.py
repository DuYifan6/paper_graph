#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Retrain ZeroGraph and positive Graph-2 for the no-static-ST robustness test.

This is a universe robustness exercise, not a point-in-time ST reconstruction.
The input panel/cache restores stocks removed by the static ST snapshot while
retaining the same observation-level tradability filter.  Both models are
retrained under the final common-MSE protocol; old TopK=60/ranking-loss weights
are never reused.

The no-static-ST common cache must itself be rebuilt with
``00_build_common_cache.py`` so imputation and normalization are fitted only
through 2024-06-05.  For the known 3,132-stock universe, disable dataset-specific
count assertions whose no-static-ST values differ from the main sample, e.g.::

    python 00_build_common_cache.py ... --expected-stocks 3132 \
        --expected-removed -1 --expected-mask-observations -1

Run examples::

    python run_nostatic_st.py --model both
    python run_nostatic_st.py --model zerograph
    python run_nostatic_st.py --model graph2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from project_config import OUTPUT_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = OUTPUT_ROOT / "no_static_st" / "common_preprocess_cache"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "no_static_st"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final-protocol no-static-ST retraining for ZeroGraph/Graph-2."
    )
    parser.add_argument("--model", choices=("both", "zerograph", "graph2"), default="both")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-stocks", type=int, default=3132)
    parser.add_argument("--sanity", action="store_true")
    return parser.parse_args(argv)


def import_script(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Required companion script not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_common_cache(cache_path: Path, meta_path: Path, expected_stocks: int) -> Dict[str, Any]:
    if not cache_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"No-static-ST common cache is missing: {cache_path} / {meta_path}"
        )
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)
    with np.load(cache_path) as pack:
        X = pack["X_np"]
        y = pack["y_np"]
        mask = pack["mask_np"]

    dates = [pd.Timestamp(value).normalize() for value in meta["all_dates"]]
    stocks = list(meta["ts_codes_order"])
    if X.shape[:2] != y.shape or y.shape != mask.shape:
        raise RuntimeError(f"Cache shape mismatch: X={X.shape}, y={y.shape}, mask={mask.shape}.")
    if X.shape[0] != len(dates) or X.shape[1] != len(stocks):
        raise RuntimeError("Cache metadata order does not match tensor dimensions.")
    if expected_stocks > 0 and len(stocks) != expected_stocks:
        raise RuntimeError(
            f"Expected {expected_stocks} no-static-ST stocks, found {len(stocks)}."
        )

    exact_fields = {
        "FIT_END": "2024-06-05",
        "VAL_START": "2024-06-06",
        "VAL_END": "2024-12-31",
        "TEST_START": "2025-01-02",
        "TEST_END": "2025-11-17",
    }
    for key, expected in exact_fields.items():
        if str(meta.get(key)) != expected:
            raise RuntimeError(
                f"No-static-ST cache is not final-protocol: {key} expected {expected}, "
                f"got {meta.get(key)!r}. Rebuild it with 00_build_common_cache.py."
            )
    prep_rule = str(meta.get("PREPROCESSING_RULE", "")).lower()
    if "fitting observations only" not in prep_rule:
        raise RuntimeError(
            "No-static-ST preprocessing is not certified fit-only. Rebuild the cache."
        )
    if meta.get("LIMIT_FILTER_ALIGNMENT") != "exact previous market trading day":
        raise RuntimeError("No-static-ST cache does not certify the final 14:30 filter alignment.")

    return {
        "stocks": len(stocks),
        "days": len(dates),
        "valid_observations": int(mask.astype(bool).sum()),
        "fit": ["2022-01-03", "2024-06-05"],
        "validation": ["2024-06-06", "2024-12-31"],
        "test": ["2025-01-02", "2025-11-17"],
        "cache_path": str(cache_path),
    }


def configure_zerograph(module, cache_path: Path, meta_path: Path, output_dir: Path, sanity: bool):
    module.CACHE_NPZ = cache_path
    module.META_PKL = meta_path
    module.OUT_DIR = output_dir
    module.RESULT_DIR = output_dir / "results"
    module.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    module.EXP_NAME = "ZeroGraph_NoStaticST_MSE"
    module.EPOCHS = 1 if sanity else 20
    module.PATIENCE = 4
    module.LR = 5e-4
    module.WEIGHT_DECAY = 1e-4


def configure_graph2(module, cache_path: Path, meta_path: Path, output_dir: Path, zero_test: Path, sanity: bool):
    module.CACHE_NPZ = str(cache_path)
    module.META_PKL = str(meta_path)
    module.PROFILE = "GRAPH2_MAIN_MSE"
    module.EXP_NAME = "Graph2_Positive_NoStaticST_MSE"
    module.ROLL_W = 20
    module.TOPK = 40
    module.EDGE_TAU = 0.10
    module.ABS_CORR = False
    module.POSITIVE_ONLY = True
    module.LAPLACIAN_LAM = 1e-3
    module.USE_RANK_LOSS = False
    module.RANK_LOSS_LAM = 0.0
    module.USE_NORMALIZED_LAPLACIAN = True
    module.EPOCHS = 20
    module.EARLY_STOP_PATIENCE = 4
    module.SANITY_ONLY = bool(sanity)
    module.ZERO_TEST_DETAIL = str(zero_test)

    asset_root = output_dir / "graph_assets"
    module.BASE_GRAPH_DIR = str(asset_root)
    module.CACHE_DIR = str(asset_root / "cache")
    module.GRAPH_DIR = str(asset_root / "graphs")
    module.FAIR_OUT_DIR = str(output_dir)
    module.RESULT_DIR = str(output_dir / "results")
    module.PLOT_DIR = str(output_dir / "plots")
    for directory in (
        module.CACHE_DIR,
        module.GRAPH_DIR,
        module.RESULT_DIR,
        module.PLOT_DIR,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)
    module.RET_MEMMAP_PATH = None
    module.RET_VALID_PATH = None
    module.GRAPH_CACHE = str(
        Path(module.GRAPH_DIR)
        / "roll20_binary_positive_mkt1_topk40_shift0_tau0.1_pow1.0_full1.npz"
    )


def collect_summary(output_dir: Path, cache_audit: Dict[str, Any]) -> None:
    rows = []
    zero_path = output_dir / "zerograph" / "results" / "summary_zerograph_msefair.csv"
    graph_path = output_dir / "graph2_positive" / "results" / "summary_graph2_mse.csv"
    if zero_path.is_file():
        row = pd.read_csv(zero_path).iloc[0].to_dict()
        rows.append(
            {
                "model": "ZeroGraph_NoStaticST_MSE",
                "test_mse": row.get("test_mse"),
                "test_r2": row.get("test_r2"),
                "best_epoch": row.get("best_epoch"),
                "best_val_mse": row.get("best_val_mse"),
            }
        )
    if graph_path.is_file():
        row = pd.read_csv(graph_path).iloc[0].to_dict()
        rows.append(
            {
                "model": "Graph2_Positive_NoStaticST_MSE",
                "test_mse": row.get("test_mse"),
                "test_r2": row.get("test_r2"),
                "best_epoch": row.get("best_epoch"),
                "best_val_mse": row.get("best_val_mse"),
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(
            output_dir / "summary_nostatic_st_models.csv",
            index=False,
            encoding="utf-8-sig",
        )
    with (output_dir / "nostatic_st_cache_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(cache_audit, handle, ensure_ascii=False, indent=2)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cache_dir = args.cache_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "common_tensor_cache.npz"
    meta_path = cache_dir / "common_tensor_meta.pkl"
    cache_audit = validate_common_cache(cache_path, meta_path, args.expected_stocks)

    zero_output = output_dir / "zerograph"
    graph_output = output_dir / "graph2_positive"
    zero_test = zero_output / "results" / "pred_zerograph_test.csv"

    if args.model in ("both", "zerograph"):
        zero = import_script(SCRIPT_DIR / "01_zerograph.py", "nostatic_zero")
        configure_zerograph(zero, cache_path, meta_path, zero_output, args.sanity)
        print("\n===== NO-STATIC-ST ZEROGRAPH =====")
        zero.main()

    if args.model in ("both", "graph2"):
        graph2 = import_script(SCRIPT_DIR / "03_graph2.py", "nostatic_graph2")
        configure_graph2(graph2, cache_path, meta_path, graph_output, zero_test, args.sanity)
        print("\n===== NO-STATIC-ST POSITIVE GRAPH-2 =====")
        graph2.main()

    collect_summary(output_dir, cache_audit)
    print(f"\nNo-static-ST outputs saved under: {output_dir}")
    if args.sanity:
        print("[WARNING] --sanity outputs are not reportable estimates.")


if __name__ == "__main__":
    main()
