import ast
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.router import route_features
from src.masking import random_masker


def load_smap_windows(data_root, machine_id, config):

    window = config['window_size'] 
    stride = config['stride'] 
    test_stride = config.get('test_stride', stride)

    print(f"Training: SMAP_{machine_id}")
    
    # 1. Loading & Standardization
    train_path = os.path.join(data_root, "train", f"{machine_id}.npy")
    test_path = os.path.join(data_root, "test", f"{machine_id}.npy")
    
    train_raw = np.load(train_path).astype(np.float32)
    test_raw = np.load(test_path).astype(np.float32)
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
        return np.array([data[i*current_stride : i*current_stride + window] for i in range(num_windows)], dtype=np.float32)

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
    csv_path = os.path.join(data_root, "labeled_anomalies.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_root, "labelled_anomalies.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SMAP Label CSV not found in {data_root}")
    df = pd.read_csv(csv_path)
    machine_rows = df[df['chan_id'] == machine_id]
    if machine_rows.empty:
        raise ValueError(f"SMAP channel {machine_id} not found in {csv_path}")

    num_test_steps = test_raw.shape[0]
    test_labels = np.zeros(num_test_steps, dtype=np.int32)
    
    for _, machine_info in machine_rows.iterrows():
        anomaly_indices = ast.literal_eval(machine_info['anomaly_sequences'])
        for start, end in anomaly_indices:
            test_labels[start : end + 1] = 1
    actual_test_len = (test_final["phy"].shape[0] - 1) * test_stride + window
    test_labels = test_labels[:actual_test_len]

    return train_final, test_final, test_labels, scaler.mean_, scaler.scale_
