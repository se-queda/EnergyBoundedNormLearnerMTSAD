import numpy as np
import os
from sklearn.preprocessing import StandardScaler

from src.router import route_features
from src.masking import random_masker
from src.utils import load_txt_file


def load_smd_windows(data_root, machine_id, config):
    window = config['window_size'] 
    stride = config['stride'] 
    test_stride = config.get('test_stride', stride)

    print(f"🚀 Dual-Anchor Pipeline Initiated: {machine_id}")
    
    # Load raw train/test points and point-level test labels.
    train_raw = load_txt_file(os.path.join(data_root, "train", f"{machine_id}.txt"))
    test_raw = load_txt_file(os.path.join(data_root, "test", f"{machine_id}.txt"))
    test_labels = load_txt_file(os.path.join(data_root, "test_label", f"{machine_id}.txt")).flatten().astype(np.int32)

    # Fit preprocessing on train only; test uses the fixed train transform.
    scaler = StandardScaler()
    train_total_norm = scaler.fit_transform(train_raw)
    test_total_norm = scaler.transform(test_raw)

    # route_features discovers topology from training statistics only.
    (train_phy, train_res, test_phy, test_res), topo, _ = route_features(
        train_total_norm, test_total_norm
    )

    def create_windows(data, current_stride):
        num_windows = (data.shape[0] - window) // current_stride + 1 
        return np.array([data[i*current_stride : i*current_stride + window] for i in range(num_windows)], dtype=np.float32)

    train_w_phy = create_windows(train_phy, stride)
    train_res_w = create_windows(train_res, stride)

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

    # Labels remain point-aligned through the last endpoint covered by windows.
    actual_test_len = (test_final["phy"].shape[0] - 1) * test_stride + window
    test_labels = test_labels[:actual_test_len]

    return train_final, test_final, test_labels, scaler.mean_, scaler.scale_
