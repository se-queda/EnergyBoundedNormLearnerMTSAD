import ast
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from src.router import route_features
from src.masking import random_masker


def _label_csv_path(data_root):
    csv_path = os.path.join(data_root, "labeled_anomalies.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_root, "labelled_anomalies.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Telemetry label CSV not found in {data_root}")
    return csv_path


def _list_channel_ids(data_root):
    train_dir = os.path.join(data_root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Telemetry train directory not found: {train_dir}")
    return sorted(f[:-4] for f in os.listdir(train_dir) if f.endswith(".npy"))


def _load_channel_arrays(data_root, machine_ids):
    train_arrays = []
    test_arrays = []
    lengths = []
    for machine_id in machine_ids:
        train_path = os.path.join(data_root, "train", f"{machine_id}.npy")
        test_path = os.path.join(data_root, "test", f"{machine_id}.npy")
        train_raw = np.load(train_path).astype(np.float32)
        test_raw = np.load(test_path).astype(np.float32)
        train_arrays.append(train_raw)
        test_arrays.append(test_raw)
        lengths.append(test_raw.shape[0])
    if not train_arrays:
        raise ValueError(f"No telemetry channels found in {data_root}")
    return train_arrays, test_arrays, lengths


def _stack_labels(data_root, machine_ids, test_lengths):
    df = pd.read_csv(_label_csv_path(data_root))
    labels = []
    for machine_id, test_len in zip(machine_ids, test_lengths):
        query = df[df["chan_id"] == machine_id]
        if query.empty:
            raise ValueError(f"Telemetry channel {machine_id} not found in labels CSV")
        channel_labels = np.zeros(test_len, dtype=np.int32)
        for _, row in query.iterrows():
            anomaly_indices = ast.literal_eval(row["anomaly_sequences"])
            for start, end in anomaly_indices:
                channel_labels[start:end + 1] = 1
        labels.append(channel_labels)
    return np.concatenate(labels, axis=0).astype(np.int32)


def _create_windows(data, window, current_stride):
    num_windows = (data.shape[0] - window) // current_stride + 1
    return np.array([data[i * current_stride:i * current_stride + window] for i in range(num_windows)], dtype=np.float32)


def _load_nasa_compact_windows(data_root, config, dataset_name):
    window = config["window_size"]
    stride = config["stride"]
    test_stride = config.get("test_stride", stride)

    print(f"Training: {dataset_name}")

    machine_ids = _list_channel_ids(data_root)
    train_arrays, test_arrays, test_lengths = _load_channel_arrays(data_root, machine_ids)
    train_raw = np.concatenate(train_arrays, axis=0).astype(np.float32)
    test_raw = np.concatenate(test_arrays, axis=0).astype(np.float32)
    test_labels = _stack_labels(data_root, machine_ids, test_lengths)

    scaler = StandardScaler()
    test_scaler = RobustScaler()
    train_total_norm = scaler.fit_transform(train_raw)
    train_robust = test_scaler.fit(train_raw)
    test_total_norm = train_robust.transform(test_raw)

    (train_phy, train_res, test_phy, test_res), topo, _ = route_features(train_total_norm, test_total_norm)

    train_w_phy = _create_windows(train_phy, window, stride)
    train_res_w = _create_windows(train_res, window, stride)

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
        "phy": _create_windows(test_phy, window, test_stride),
        "res": _create_windows(test_res, window, test_stride),
        "topology": topo,
    }

    actual_test_len = (test_final["phy"].shape[0] - 1) * test_stride + window
    test_labels = test_labels[:actual_test_len]

    return train_final, test_final, test_labels, scaler.mean_, scaler.scale_


def load_smap_compact_windows(data_root, config):
    return _load_nasa_compact_windows(data_root, config, "SMAP")


def load_msl_compact_windows(data_root, config):
    return _load_nasa_compact_windows(data_root, config, "MSL")
