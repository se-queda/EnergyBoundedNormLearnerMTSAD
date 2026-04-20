import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Internal imports
from src.router import route_features
from src.masking import random_masker


def _infer_time_columns(df):
    time_cols = []
    first_col = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[first_col]):
        time_cols.append(first_col)
    for col in df.columns:
        name = col.strip().lower()
        if "timestamp" in name or name == "time":
            if col not in time_cols:
                time_cols.append(col)
    return time_cols


def _coerce_features(df, use_bfill):
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.ffill()
    if use_bfill:
        numeric_df = numeric_df.bfill()
    return numeric_df.values.astype(np.float32)


def _labels_from_series(series):
    if series is None:
        return None
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(np.int32).values
    values = series.astype(str).str.strip().str.lower()
    is_attack = values.str.contains("attack|anomaly|true") | (values == "1")
    return is_attack.astype(np.int32).values


def _has_bad_header(df):
    cols = [str(c).strip().lower() for c in df.columns]
    if not cols:
        return True
    unnamed = sum(c.startswith("unnamed") for c in cols)
    return unnamed >= max(1, len(cols) // 2)


def _read_table(path, use_bfill):
    csv_path = os.path.splitext(path)[0] + ".csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if _has_bad_header(df):
            df = pd.read_excel(path, header=1).ffill()
            df.to_csv(csv_path, index=False)
    else:
        df = pd.read_excel(path).ffill()
        if _has_bad_header(df):
            df = pd.read_excel(path, header=1).ffill()
        df.to_csv(csv_path, index=False)
    if use_bfill:
        df = df.bfill()
    return df


def load_swat_windows(data_root, config):
    if isinstance(data_root, dict):
        if "SWAT" in data_root:
            data_root = data_root["SWAT"]
        else:
            data_root = next(iter(data_root.values()), None)

    window = config["window_size"]
    stride = config["stride"]
    test_stride = config.get("test_stride", stride)

    print("Training: SWAT Dataset")
    # 1. Loading & Standardization
    train_path = os.path.join(data_root, "train.xlsx")
    test_path = os.path.join(data_root, "test.xlsx")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"SWAT train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"SWAT test file not found: {test_path}")

    train_df = _read_table(train_path, use_bfill=True)
    test_df = _read_table(test_path, use_bfill=False)

    train_df.columns = [str(c).strip() for c in train_df.columns]
    test_df.columns = [str(c).strip() for c in test_df.columns]

    def _canon(name):
        return re.sub(r"\s+", "", str(name)).lower()

    label_col = None
    for c in test_df.columns:
        if _canon(c) == "normal/attack":
            label_col = c
            break
    if label_col is None:
        raise ValueError("SWAT Test.xlsx must include a 'Normal/Attack' column. Columns: " + str(list(test_df.columns)[:10]))

    test_labels = _labels_from_series(test_df[label_col])
    test_df = test_df.drop(columns=[label_col])
    if label_col in train_df.columns:
        train_df = train_df.drop(columns=[label_col])

    for col in _infer_time_columns(train_df):
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
    for col in _infer_time_columns(test_df):
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])

    train_map = {_canon(c): c for c in train_df.columns}
    test_map = {_canon(c): c for c in test_df.columns}
    common_keys = [k for k in train_map.keys() if k in test_map]
    if not common_keys:
        raise ValueError(
            "SWAT train/test files have no common feature columns. "
            f"train sample={list(train_df.columns[:8])} test sample={list(test_df.columns[:8])}"
        )
    train_df = train_df[[train_map[k] for k in common_keys]]
    test_df = test_df[[test_map[k] for k in common_keys]]
    train_df.columns = common_keys
    test_df.columns = common_keys

    train_raw = _coerce_features(train_df, use_bfill=True)
    test_raw = _coerce_features(test_df, use_bfill=False)
    scaler = StandardScaler()
    train_total_norm = scaler.fit_transform(train_raw)
    test_total_norm = scaler.transform(test_raw)

    # 2. Routing
    (train_phy, train_res, test_phy, test_res), topo, _ = route_features(
        train_total_norm, test_total_norm
    )

    # 3. Windowing
    def create_windows(data, current_stride):
        num_windows = (data.shape[0] - window) // current_stride + 1
        return np.array(
            [data[i * current_stride : i * current_stride + window] for i in range(num_windows)],
            dtype=np.float32,
        )

    train_w_phy = create_windows(train_phy, stride)
    train_res_w = create_windows(train_res, stride)

    # 4. Masking Views
    v1, v2, v3, v4 = random_masker(train_w_phy)
    phy_views = np.stack([train_w_phy, v1, v2, v3, v4], axis=1)
    rv1, = random_masker(train_res_w, mask_rates=(0.25,))

    train_final = {
        "phy_views": phy_views,
        "res_views": rv1,
        "phy_anchor": train_w_phy,
        "res_orig": train_res_w,
        "topology": topo,
    }

    test_final = {
        "phy": create_windows(test_phy, test_stride),
        "res": create_windows(test_res, test_stride),
        "topology": topo,
    }

    # 5.Label Parsing 
    actual_test_len = (test_final["phy"].shape[0] - 1) * test_stride + window
    if test_labels.size < actual_test_len:
        raise ValueError(
            f"SWAT labels shorter than expected: {test_labels.size} < {actual_test_len}"
        )
    test_labels = test_labels[:actual_test_len]
    return train_final, test_final, test_labels, scaler.mean_, scaler.scale_
