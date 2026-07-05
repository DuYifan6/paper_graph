# -*- coding: utf-8 -*-
"""
从你现有的 common_tensor_cache.npz / common_tensor_meta.pkl
导出 SOTA 基准实验需要的数据格式。

你可以直接在 PyCharm 里运行本脚本，不需要命令行参数。

输出目录会生成：
    X.npy
    y.npy
    valid_mask.npy
    dates.npy
    stocks.npy
    meta.json
    graph2_edges.npz   # 如果能自动识别你的 GRAPH_CACHE，则会生成；否则只打印 graph keys

然后运行：
    run_sota_representative_baselines.py
"""

import os
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# =======================
# 0) 直接改这里即可
# =======================
CACHE_NPZ = r"D:\PythonProject\LASSO_FINAL\common_preprocess_cache\common_tensor_cache.npz"
META_PKL  = r"D:\PythonProject\LASSO_FINAL\common_preprocess_cache\common_tensor_meta.pkl"

OUT_DIR = r"D:\PythonProject\LASSO_FINAL\SOTA\tensor_cache_noST_no9pct"

# 你的 Graph-2 已经构好的图缓存；非图 SOTA 不需要它，GCN-LSTM 才需要
GRAPH_CACHE = r"D:\PythonProject\LASSO_FINAL\graph_调整\graph2_binary_60\graphs\roll20_binary_abs1_mkt1_topk60_shift0_tau0.1_pow1.5_full1.npz"

# 论文最终 9 个稳定特征
STABLE_FEATURES = [
    "M_RSI_6_5",
    "D_pct_chg",
    "M_MACD_1",
    "D_mainforce_strength",
    "D_bull_bear_ratio",
    "D_intraday_return",
    "M_high_1",
    "D_overnight_return",
    "M_MACD_hist_1",
]


# =======================
# 1) 通用工具
# =======================
def load_meta(meta_path):
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def show_npz_keys(npz_path):
    print("\n" + "=" * 80)
    print(f"检查 NPZ: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    print("keys =", z.files)
    for k in z.files:
        arr = z[k]
        try:
            print(f"  {k:30s} shape={arr.shape}, dtype={arr.dtype}")
        except Exception:
            print(f"  {k:30s} type={type(arr)}")
    return z


def show_meta(meta):
    print("\n" + "=" * 80)
    print("检查 META")
    print("meta type =", type(meta))
    if isinstance(meta, dict):
        print("meta keys =", list(meta.keys()))
        for k, v in meta.items():
            if isinstance(v, (list, tuple, np.ndarray, pd.Index)):
                print(f"  {k:30s} len={len(v)} sample={list(v)[:5]}")
            elif isinstance(v, pd.DataFrame):
                print(f"  {k:30s} DataFrame shape={v.shape}, cols={list(v.columns)[:8]}")
            else:
                print(f"  {k:30s} value={repr(v)[:120]}")
    else:
        attrs = [a for a in dir(meta) if not a.startswith("_")]
        print("attrs sample =", attrs[:50])
        for k in attrs[:50]:
            try:
                v = getattr(meta, k)
                if isinstance(v, (list, tuple, np.ndarray, pd.Index)):
                    print(f"  {k:30s} len={len(v)} sample={list(v)[:5]}")
                else:
                    print(f"  {k:30s} type={type(v)} repr={repr(v)[:80]}")
            except Exception:
                pass


def get_from_meta(meta, candidates):
    """
    从 dict 或对象属性中找候选键。
    """
    if isinstance(meta, dict):
        for k in candidates:
            if k in meta:
                return meta[k], k
    else:
        for k in candidates:
            if hasattr(meta, k):
                return getattr(meta, k), k
    return None, None


def get_from_npz(z, candidates):
    for k in candidates:
        if k in z.files:
            return z[k], k
    return None, None


def find_first_array_by_shape(z, ndim=None, shape_tail=None):
    for k in z.files:
        arr = z[k]
        if ndim is not None and arr.ndim != ndim:
            continue
        if shape_tail is not None and tuple(arr.shape[-len(shape_tail):]) != tuple(shape_tail):
            continue
        return arr, k
    return None, None


def normalize_dates(x):
    x = np.asarray(x)
    # 处理 pandas datetime、字符串日期、20250103 数字日期
    if np.issubdtype(x.dtype, np.number):
        out = pd.to_datetime(x.astype(int).astype(str), errors="coerce")
    else:
        out = pd.to_datetime(x.astype(str), errors="coerce")
    if out.isna().any():
        bad = np.where(out.isna())[0][:10]
        raise ValueError(f"dates 中存在无法解析的日期，位置示例: {bad}")
    return np.array([pd.Timestamp(d).strftime("%Y-%m-%d") for d in out], dtype=object)


def normalize_stocks(x):
    return np.asarray(x).astype(str)


# =======================
# 2) 读取 common cache，自动识别 X/y/valid/dates/stocks
# =======================
def export_tensor_cache():
    os.makedirs(OUT_DIR, exist_ok=True)

    z = show_npz_keys(CACHE_NPZ)
    meta = load_meta(META_PKL)
    show_meta(meta)

    # ---- X: 期望 [T, N, F]
    X_candidates = [
        "X", "x", "X_all", "features", "feature_tensor", "X_tensor",
        "X_np", "panel_X", "tensor_X"
    ]
    X, x_key = get_from_npz(z, X_candidates)

    if X is None:
        # 自动找第一个 3 维数组
        X, x_key = find_first_array_by_shape(z, ndim=3)

    if X is None:
        raise KeyError("没有在 common_tensor_cache.npz 里找到 3维 X 数组。请把上面打印的 keys 发给我。")

    X = np.asarray(X)
    print(f"\n识别到 X: key={x_key}, shape={X.shape}, dtype={X.dtype}")

    # ---- y: 期望 [T, N]
    y_candidates = [
        "y", "Y", "target", "targets", "gap_up_pct", "gap", "overnight_gap_return",
        "gap_up", "label", "labels"
    ]
    y, y_key = get_from_npz(z, y_candidates)

    if y is None:
        # 自动找与 X 前两维一致的二维数组
        for k in z.files:
            arr = z[k]
            if arr.ndim == 2 and tuple(arr.shape) == tuple(X.shape[:2]):
                y, y_key = arr, k
                break

    if y is None:
        raise KeyError("没有在 common_tensor_cache.npz 里找到 y 数组。请把上面打印的 keys 发给我。")

    y = np.asarray(y)
    print(f"识别到 y: key={y_key}, shape={y.shape}, dtype={y.dtype}")

    # ---- valid mask: 期望 [T, N]
    valid_candidates = [
        "valid_mask", "mask", "valid", "valid_y", "valid_panel",
        "is_valid", "obs_mask", "mask_y"
    ]
    valid, valid_key = get_from_npz(z, valid_candidates)

    if valid is not None:
        valid = np.asarray(valid).astype(bool)
        print(f"识别到 valid_mask: key={valid_key}, shape={valid.shape}, dtype={valid.dtype}")
        if tuple(valid.shape) != tuple(y.shape):
            print(f"警告：valid_mask shape={valid.shape} 与 y shape={y.shape} 不一致，将重新计算 valid_mask。")
            valid = None

    if valid is None:
        valid = np.isfinite(y) & np.isfinite(X).all(axis=2)
        valid_key = "computed_from_finite_X_y"
        print(f"没有找到可用 valid_mask，已根据 finite X/y 重新计算: shape={valid.shape}")

    # ---- dates
    date_candidates = [
        "dates", "date_list", "trade_dates", "trading_days", "all_dates",
        "unique_dates", "date_index"
    ]
    dates, date_key = get_from_npz(z, date_candidates)
    if dates is None:
        dates, date_key = get_from_meta(meta, date_candidates)

    if dates is None:
        raise KeyError("没有找到 dates/trade_dates。请把 META 打印信息发给我。")

    dates = normalize_dates(dates)
    print(f"识别到 dates: key={date_key}, len={len(dates)}, {dates[0]} .. {dates[-1]}")

    # ---- stocks
    stock_candidates = [
        "stocks", "stock_list", "ts_codes", "ts_code_list", "code_list",
        "codes", "tickers", "symbols", "stock_codes", "stock_ids",
        "all_stocks", "unique_stocks", "stock_index", "columns_stock",
        "stock_universe", "universe", "valid_stocks"
    ]
    stocks, stock_key = get_from_npz(z, stock_candidates)
    if stocks is None:
        stocks, stock_key = get_from_meta(meta, stock_candidates)

    # 对 SOTA 基准实验来说，真实股票代码不是必须的；只要节点顺序与 X/y 一致即可。
    # 如果 common meta 里没有股票代码，就用 0..N-1 作为节点 ID，后续所有模型和指标仍然有效。
    if stocks is None:
        stocks = np.array([str(i) for i in range(X.shape[1])], dtype=object)
        stock_key = "generated_0_to_N_minus_1"
        print("\n警告：没有在 meta/npz 中找到 stocks/ts_codes。")
        print("已自动生成股票节点 ID：'0', '1', ..., 'N-1'。")
        print("这不影响 Ridge/LightGBM/MLP/Transformer/GCN-LSTM 的训练和指标计算。")
        print("如果后面要和你已有 ZeroGraph/Graph2 CSV 按真实 ts_code 合并，再需要补真实股票代码。")

    stocks = normalize_stocks(stocks)
    print(f"识别到 stocks: key={stock_key}, len={len(stocks)}, sample={stocks[:5]}")

    # ---- feature names
    feature_name_candidates = [
        "feature_cols", "features", "feature_names", "x_cols",
        "selected_features", "stable_features", "cols_x"
    ]
    feature_names, feature_key = get_from_npz(z, feature_name_candidates)
    if feature_names is None:
        feature_names, feature_key = get_from_meta(meta, feature_name_candidates)

    if feature_names is not None:
        feature_names = [str(c) for c in list(feature_names)]
        print(f"识别到 feature_names: key={feature_key}, len={len(feature_names)}, sample={feature_names[:10]}")
    else:
        feature_names = None
        print("没有识别到 feature_names。")

    # ---- 只保留 9 个 stable features
    if X.shape[2] == len(STABLE_FEATURES):
        print("\nX 的 F=9，认为已经是 stable feature tensor，不再筛选特征。")
        selected_features = STABLE_FEATURES
        X_out = X.astype(np.float32)
    else:
        if feature_names is None:
            raise ValueError(
                f"X.shape[2]={X.shape[2]}，不是 9；但没有 feature_names，无法筛选 9 个稳定特征。\n"
                "请把上面打印的 meta keys 发给我，或者手动确认 common cache 是否已经是 9 特征。"
            )

        missing = [c for c in STABLE_FEATURES if c not in feature_names]
        if missing:
            raise KeyError(
                f"feature_names 中找不到这些 stable features: {missing}\n"
                f"feature_names sample={feature_names[:80]}"
            )

        idx = [feature_names.index(c) for c in STABLE_FEATURES]
        X_out = X[:, :, idx].astype(np.float32)
        selected_features = STABLE_FEATURES
        print(f"\nX 原始 F={X.shape[2]}，已筛选为 9 个 stable features: idx={idx}")

    y_out = y.astype(np.float32)
    valid_out = valid.astype(bool)

    # ---- shape 检查
    T, N, F = X_out.shape
    assert y_out.shape == (T, N), f"y shape={y_out.shape}, X={X_out.shape}"
    assert valid_out.shape == (T, N), f"valid shape={valid_out.shape}, X={X_out.shape}"
    assert len(dates) == T, f"len(dates)={len(dates)}, T={T}"
    assert len(stocks) == N, f"len(stocks)={len(stocks)}, N={N}"

    # ---- 保存
    np.save(os.path.join(OUT_DIR, "X.npy"), X_out)
    np.save(os.path.join(OUT_DIR, "y.npy"), y_out)
    np.save(os.path.join(OUT_DIR, "valid_mask.npy"), valid_out)
    np.save(os.path.join(OUT_DIR, "dates.npy"), dates)
    np.save(os.path.join(OUT_DIR, "stocks.npy"), stocks)

    out_meta = {
        "source_CACHE_NPZ": CACHE_NPZ,
        "source_META_PKL": META_PKL,
        "X_key": x_key,
        "y_key": y_key,
        "valid_key": valid_key,
        "date_key": date_key,
        "stock_key": stock_key,
        "feature_key": feature_key,
        "T": int(T),
        "N": int(N),
        "F": int(F),
        "date_start": str(dates[0]),
        "date_end": str(dates[-1]),
        "selected_features": selected_features,
        "valid_obs": int(valid_out.sum()),
        "valid_ratio": float(valid_out.mean()),
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(out_meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("SOTA tensor cache 已保存：")
    print(f"OUT_DIR     = {OUT_DIR}")
    print(f"X.npy       = {X_out.shape}")
    print(f"y.npy       = {y_out.shape}")
    print(f"valid_mask  = {valid_out.shape}, valid_obs={valid_out.sum():,}")
    print(f"dates       = {dates[0]} .. {dates[-1]}")
    print(f"stocks      = {len(stocks):,}")
    print(f"features    = {selected_features}")

    return T, N


# =======================
# 3) 可选：尝试把 Graph-2 图缓存转成 SOTA 图基线格式
# =======================
def try_export_graph_edges(T, N):
    if not GRAPH_CACHE or not os.path.exists(GRAPH_CACHE):
        print("\n没有找到 GRAPH_CACHE，跳过 graph2_edges.npz。非图基线不受影响。")
        return

    z = show_npz_keys(GRAPH_CACHE)
    keys = set(z.files)
    out_path = os.path.join(OUT_DIR, "graph2_edges.npz")

    # 如果已经是目标格式，直接复制/另存
    if "edge_index_by_t" in keys:
        edge_index_by_t = z["edge_index_by_t"]
        edge_weight_by_t = z["edge_weight_by_t"] if "edge_weight_by_t" in keys else None
        if edge_weight_by_t is None:
            edge_weight_by_t = np.empty(len(edge_index_by_t), dtype=object)
            for i in range(len(edge_index_by_t)):
                edge_weight_by_t[i] = np.ones(edge_index_by_t[i].shape[1], dtype=np.float32)
        np.savez(out_path, edge_index_by_t=edge_index_by_t, edge_weight_by_t=edge_weight_by_t)
        print(f"\n已导出 graph2_edges.npz: {out_path}")
        return

    # 常见格式：每天 rows/cols/weights
    row_key_candidates = ["rows_by_t", "row_by_t", "edge_rows_by_t", "src_by_t", "sources_by_t"]
    col_key_candidates = ["cols_by_t", "col_by_t", "edge_cols_by_t", "dst_by_t", "targets_by_t"]
    weight_key_candidates = ["weights_by_t", "edge_weight_by_t", "vals_by_t", "data_by_t", "edge_weights_by_t"]

    row_key = next((k for k in row_key_candidates if k in keys), None)
    col_key = next((k for k in col_key_candidates if k in keys), None)
    w_key = next((k for k in weight_key_candidates if k in keys), None)

    if row_key and col_key:
        rows_by_t = z[row_key]
        cols_by_t = z[col_key]
        weights_by_t = z[w_key] if w_key else None

        edge_index_by_t = np.empty(len(rows_by_t), dtype=object)
        edge_weight_by_t = np.empty(len(rows_by_t), dtype=object)

        for t in range(len(rows_by_t)):
            r = np.asarray(rows_by_t[t], dtype=np.int64)
            c = np.asarray(cols_by_t[t], dtype=np.int64)
            edge_index_by_t[t] = np.vstack([r, c])
            if weights_by_t is not None:
                edge_weight_by_t[t] = np.asarray(weights_by_t[t], dtype=np.float32)
            else:
                edge_weight_by_t[t] = np.ones(len(r), dtype=np.float32)

        np.savez(out_path, edge_index_by_t=edge_index_by_t, edge_weight_by_t=edge_weight_by_t)
        print(f"\n已从 {row_key}/{col_key} 导出 graph2_edges.npz: {out_path}")
        return

    # CSR object arrays
    if "indices_by_t" in keys and "indptr_by_t" in keys:
        # SOTA runner 本身也支持 CSR，但这里仍另存一份统一格式。
        indices_by_t = z["indices_by_t"]
        indptr_by_t = z["indptr_by_t"]
        data_by_t = z["data_by_t"] if "data_by_t" in keys else None

        edge_index_by_t = np.empty(len(indices_by_t), dtype=object)
        edge_weight_by_t = np.empty(len(indices_by_t), dtype=object)

        for t in range(len(indices_by_t)):
            indices = np.asarray(indices_by_t[t], dtype=np.int64)
            indptr = np.asarray(indptr_by_t[t], dtype=np.int64)
            rows = []
            cols = []
            vals = []
            for r in range(len(indptr) - 1):
                s, e = indptr[r], indptr[r + 1]
                if e > s:
                    c = indices[s:e]
                    rows.append(np.full(len(c), r, dtype=np.int64))
                    cols.append(c)
                    if data_by_t is not None:
                        vals.append(np.asarray(data_by_t[t][s:e], dtype=np.float32))
                    else:
                        vals.append(np.ones(len(c), dtype=np.float32))

            if rows:
                edge_index_by_t[t] = np.vstack([np.concatenate(rows), np.concatenate(cols)])
                edge_weight_by_t[t] = np.concatenate(vals)
            else:
                edge_index_by_t[t] = np.zeros((2, 0), dtype=np.int64)
                edge_weight_by_t[t] = np.zeros(0, dtype=np.float32)

        np.savez(out_path, edge_index_by_t=edge_index_by_t, edge_weight_by_t=edge_weight_by_t)
        print(f"\n已从 CSR 格式导出 graph2_edges.npz: {out_path}")
        return

    print("\n没有自动识别出 Graph cache 的边格式。")
    print("这不影响先跑 ridge/lightgbm/mlp/transformer。")
    print("请把上面 GRAPH_CACHE 打印出来的 keys 发给我，我再按你的实际格式改转换代码。")


if __name__ == "__main__":
    T, N = export_tensor_cache()
    try_export_graph_edges(T, N)

    print("\n下一步可运行非图代表性基线：")
    print(rf'D:\Anaconda3\python.exe D:\PythonProject\LASSO_FINAL\SOTA\run_sota_representative_baselines.py --data_dir "{OUT_DIR}" --out_dir "D:\PythonProject\LASSO_FINAL\SOTA\sota_results_rep" --models ridge,lightgbm,mlp,transformer --lookback 20 --epochs 15 --batch_size 4096')

    print("\n如果 graph2_edges.npz 已生成，再运行 GCN-LSTM：")
    print(rf'D:\Anaconda3\python.exe D:\PythonProject\LASSO_FINAL\SOTA\run_sota_representative_baselines.py --data_dir "{OUT_DIR}" --graph_edges "{os.path.join(OUT_DIR, "graph2_edges.npz")}" --out_dir "D:\PythonProject\LASSO_FINAL\SOTA\sota_results_rep_graph" --models gcn_lstm --lookback 20 --graph_epochs 15')
