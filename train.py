import yaml
import argparse
import os
import numpy as np
import gc
from tensorflow.keras import backend as K
import tensorflow as tf
import traceback
import time
from tqdm import tqdm
from src.data_loaders.psmloader import load_psm_windows
from src.data_loaders.smdloader import load_smd_windows
from src.data_loaders.smd_compact_loader import load_smd_compact_windows
from src.data_loaders.smaploader import load_smap_windows
from src.data_loaders.msloader import load_msl_windows
from src.data_loaders.swatloader import load_swat_windows
from scripts.generalized_anomaly_scorer import score_raw_entity
# from src.data_loaders.swatloader import load_swat_windows


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _resolve_data_root(config, dataset):
    data_root = config.get("data_root")
    data_roots = config.get("data_roots")
    if isinstance(data_roots, dict) and (data_root is None or isinstance(data_root, dict)):
        data_root = data_roots

    if isinstance(data_root, dict):
        candidates = [
            dataset,
            dataset.upper(),
            dataset.lower(),
            dataset.title(),
        ]
        for key in candidates:
            if key in data_root:
                return data_root[key]
        return next(iter(data_root.values()), None)
    return data_root

def train_on_machine(machine_id, config):
    dataset_type = config.get("dataset", "SMD").upper()
    print(f"\n🚀 Dual-Anchor Engine Deployment [{dataset_type}]: {machine_id}")

    # Standardized Parameters
    W = config["window_size"] 
    BS = config["batch_size"] 
    TEST_BS = int(config.get("test_batch_size", BS))
    VS = config["val_split"]  
    EP = config["epochs"]     
    L = config["latent_dim"]  
    stride = config["stride"]
    test_stride = int(config.get("test_stride", stride))
    # 1. Dynamic Loader Selection 
    data_root = _resolve_data_root(config, dataset_type)
    if not data_root:
        raise ValueError(f"data_root not set for dataset {dataset_type}")
    if dataset_type == "PSM":
        train_final, test_final, test_labels, _, _ = load_psm_windows(data_root, config)
    elif dataset_type == "SMAP":
        train_final, test_final, test_labels, _, _ = load_smap_windows(data_root, machine_id, config)
    elif dataset_type == "MSL":
        train_final, test_final, test_labels, _, _ = load_msl_windows(data_root, machine_id, config)
    elif dataset_type == "SWAT":
          train_final, test_final, test_labels, _, _ = load_swat_windows(data_root, config)
    else:
        if config.get("smd_compact", False):
            train_final, test_final, test_labels, _, _ = load_smd_compact_windows(data_root, config)
        else:
            train_final, test_final, test_labels, _, _ = load_smd_windows(data_root, machine_id, config)
    
    topo = train_final['topology']
    phy_dim = len(topo.idx_phy)
    res_dim = len(topo.idx_res)
    
    # DYNAMIC PATIENCE SELECTION
    total_sensors = len(topo.idx_phy) + len(topo.idx_res)
    base_patience = int(config.get("patience", 10))
    if len(topo.res_to_dead_local) == total_sensors and total_sensors > 0:
        # All sensors are dead
        current_patience = 100 
        print(f"📡 Mode: All Sensors Dead. Setting High Patience ({current_patience})")
    else:
        # Healthy Machine: low patience prevents overfitting
        current_patience = base_patience
        print(f"🛰️ Mode: Dual-Anchor. Setting Standard Patience ({current_patience})")
    # 2. Build Datasets
    from src.utils import build_tf_datasets
    train_ds, val_ds, _, train_idx, val_idx = build_tf_datasets(
        train_final,
        test_final,
        val_split=VS,
        batch_size=BS,
        val_normal_only=config.get("val_normal_only", True),
        window_size=W,
        stride=stride,
    )

    # 3. Build Models
    from src.models import build_dual_encoder, build_dual_decoder, build_discriminator, build_res_discriminator
    encoder = build_dual_encoder(input_shape_sys=(W, phy_dim), input_shape_res=(W, res_dim + 1), config=config)
    decoder = build_dual_decoder(feat_sys=phy_dim, feat_res=res_dim, output_steps=W, config=config)
    discriminator = build_discriminator(input_dim=L)
    res_discriminator = build_res_discriminator(input_dim=L // 2)

    # 4. Trainer Initialization
    from src.trainer import EBNL_Trainer
    trainer = EBNL_Trainer(
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator,
        res_discriminator=res_discriminator,
        config={**config, "patience": current_patience},
        topology=topo,
    )
    
    # Training with Early Stopping
    train_stats = trainer.fit(train_ds, val_ds=val_ds, epochs=EP)
    
    # Save Weights
    save_path = os.path.join("weights", dataset_type, str(machine_id))
    os.makedirs(save_path, exist_ok=True)
    trainer.encoder.save_weights(f"{save_path}/encoder.weights.h5")
    trainer.decoder.save_weights(f"{save_path}/decoder.weights.h5")
    # 5. Save raw reconstruction artifacts for the fixed external scorer
    recons = trainer.reconstruct(test_final, batch_size=TEST_BS)
    actual_len = (recons['res_orig'].shape[0] - 1) * test_stride + W
    res_orig, res_hat = recons['res_orig'], recons['res_hat']

    def _compute_train_raw_artifacts():
        if train_idx is None or len(train_idx) == 0 or "phy_anchor" not in train_final or "res_orig" not in train_final:
            return None, None, None
        train_order = np.sort(train_idx)
        train_phy = train_final["phy_anchor"][train_order]
        train_res = train_final["res_orig"][train_order]
        train_idx_shifted = train_order
        num_train_windows = len(train_final["phy_anchor"])
        if len(train_idx_shifted) > 0:
            if train_idx_shifted.min() < 0 or train_idx_shifted.max() > num_train_windows - 1:
                raise RuntimeError(
                    f"Train window index out of bounds: expected within [0,{num_train_windows-1}] "
                    f"got [{train_idx_shifted.min()},{train_idx_shifted.max()}]."
                )
            train_total_len = (num_train_windows - 1) * stride + W
        else:
            train_total_len = 0

        if train_total_len <= 0:
            return None, None, None

        train_recons = trainer.reconstruct({"phy": train_phy, "res": train_res})
        return train_recons, train_idx_shifted, train_total_len

    train_recons, train_idx_shifted, train_total_len = _compute_train_raw_artifacts()
    if train_recons is None:
        raise RuntimeError("Train raw artifacts unavailable for external scoring.")

    test_labels = test_labels[:actual_len]
    
    print(f" Raw windows stitched to {actual_len} points | Labels: {len(test_labels)} | Anomaly Rate: {np.mean(test_labels):.2%}")

    test_phy_raw = None
    test_res_raw = None
    if phy_dim > 0:
        test_phy_raw = np.square(recons["phy_orig"] - recons["phy_hat"])
    if res_orig.shape[-1] > 0:
        test_res_raw = np.square(res_orig - res_hat)

    train_phy_raw = None
    train_res_raw = None
    if train_recons is not None and phy_dim > 0:
        train_phy_raw = np.square(train_recons["phy_orig"] - train_recons["phy_hat"])
    if train_recons is not None and train_recons["res_orig"].shape[-1] > 0:
        train_res_raw = np.square(train_recons["res_orig"] - train_recons["res_hat"])

    def _save_raw_feature_scores(out_dir, split, phy_err, res_err, window_indices, total_len, stride_val, window_size, labels=None):
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, f"{split}_meta.npz"),
            window_indices=np.asarray(window_indices) if window_indices is not None else None,
            total_len=int(total_len),
            stride=int(stride_val),
            window_size=int(window_size),
            labels=np.asarray(labels) if labels is not None else None,
        )
        if phy_err is not None:
            np.savez_compressed(
                os.path.join(out_dir, f"{split}_phy_raw.npz"),
                err=np.asarray(phy_err),
            )
        if res_err is not None:
            np.savez_compressed(
                os.path.join(out_dir, f"{split}_res_raw.npz"),
                err=np.asarray(res_err),
            )

    if config.get("saved_score", False):
        scores_dir = os.path.join("scores", dataset_type, str(machine_id), "raw")
        os.makedirs(scores_dir, exist_ok=True)
        _save_raw_feature_scores(
            scores_dir,
            "test",
            test_phy_raw,
            test_res_raw,
            None,
            actual_len,
            test_stride,
            W,
            labels=test_labels,
        )
        _save_raw_feature_scores(
            scores_dir,
            "train",
            train_phy_raw,
            train_res_raw,
            train_idx_shifted,
            train_total_len,
            stride,
            W,
            labels=None,
        )

    score_out_dir = os.path.join(
        "posthoc_runs",
        dataset_type,
        str(machine_id),
        "generalized_anomaly_scorer",
    )
    scored = score_raw_entity(
        dataset=dataset_type,
        entity_id=machine_id,
        labels=test_labels,
        test_phy=test_phy_raw,
        test_res=test_res_raw,
        train_phy=train_phy_raw,
        train_res=train_res_raw,
        test_window_size=W,
        test_stride=test_stride,
        test_total_len=actual_len,
        train_window_size=W,
        train_stride=stride,
        train_total_len=train_total_len,
        test_window_indices=None,
        train_window_indices=train_idx_shifted,
        out_dir=score_out_dir,
    )

    auc_score = float(scored["auc"])
    pr_auc_score = float(scored["prauc"])
    p_best = float(scored["p_best"])
    r_best = float(scored["r_best"])
    f1_best = float(scored["f1_best"])
    vusauc = float(scored["vusaucc"])
    vuspr = float(scored["vuspr"])
    aff_p = float(scored["aff_p"])
    aff_r = float(scored["aff_r"])
    aff1 = float(scored["aff1"])
    print(f"[RESULTS] AUC: {auc_score:.4f} | PR-AUC: {pr_auc_score:.4f} | P: {p_best:.4f} | R: {r_best:.4f} | F1: {f1_best:.4f} | vusauc: {vusauc:.4f} | vuspr: {vuspr:.4f} | aff_p: {aff_p:.4f} | aff_r: {aff_r:.4f} | aff1: {aff1:.4f}")
    K.clear_session()

    gc.collect()

    return (
        auc_score, pr_auc_score, p_best, r_best, f1_best, vusauc, vuspr, aff_p, aff_r, aff1,
        train_stats, (encoder, decoder, discriminator, res_discriminator)
    )
