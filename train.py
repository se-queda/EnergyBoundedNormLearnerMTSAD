import yaml
import os
import numpy as np
import gc
import time
from tensorflow.keras import backend as K
import tensorflow as tf
from tqdm import tqdm
from diagnosis import compute_diagnosis_metrics
from inference import summarize_inference
from parameter import summarize_parameters
from src.data_loaders.psmloader import load_psm_windows
from src.data_loaders.smdloader import load_smd_windows
from src.data_loaders.smd_compact_loader import load_smd_compact_windows
from src.data_loaders.smaploader import load_smap_windows
from src.data_loaders.msloader import load_msl_windows
from src.data_loaders.nasa_compact_loader import load_smap_compact_windows, load_msl_compact_windows
from src.data_loaders.swatloader import load_swat_windows
from scripts.generalized_anomaly_scorer import score_stitched_entity
from scripts.legacy_scorer import score_legacy_stitched_entity


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
    W = config["window_size"] 
    BS = config["batch_size"] 
    TEST_BS = int(config.get("test_batch_size", BS))
    VS = config["val_split"]  
    EP = config["epochs"]     
    L = config["latent_dim"]  
    stride = config["stride"]
    test_stride = int(config.get("test_stride", stride))
    data_root = _resolve_data_root(config, dataset_type)
    if not data_root:
        raise ValueError(f"data_root not set for dataset {dataset_type}")
    if dataset_type == "PSM":
        train_final, test_final, test_labels, _, _ = load_psm_windows(data_root, config)
    elif dataset_type == "SMAP":
        if config.get("smap_compact", False):
            train_final, test_final, test_labels, _, _ = load_smap_compact_windows(data_root, config)
        else:
            train_final, test_final, test_labels, _, _ = load_smap_windows(data_root, machine_id, config)
    elif dataset_type == "MSL":
        if config.get("msl_compact", False):
            train_final, test_final, test_labels, _, _ = load_msl_compact_windows(data_root, config)
        else:
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
    
    total_sensors = len(topo.idx_phy) + len(topo.idx_res)
    base_patience = int(config.get("patience", 10))
    if len(topo.res_to_dead_local) == total_sensors and total_sensors > 0:
        # All sensors are dead
        current_patience = 100 
        print(f"All Sensors Dead. Setting High Patience ({current_patience})")
    else:
        # Healthy Machine low patience prevents overfitting
        current_patience = base_patience
        print(f"Setting Standard Patience ({current_patience})")
        
    # 2. Build Datasets
    from src.utils import build_tf_datasets
    train_ds, val_ds, train_idx, val_idx = build_tf_datasets(
        train_final,
        val_split=VS,
        batch_size=BS,
        val_normal_only=config.get("val_normal_only", True),
        window_size=W,
        stride=stride,
    )

    # 3. Build Models
    from src.models import build_dual_encoder, build_sys_decoder, build_res_decoder, build_discriminator, build_res_discriminator
    encoder = build_dual_encoder(input_shape_sys=(W, phy_dim), input_shape_res=(W, res_dim + 1), config=config)
    decoder_sys = build_sys_decoder(feat_sys=phy_dim, output_steps=W, config=config)
    decoder_res = build_res_decoder(feat_res=res_dim, output_steps=W, config=config)
    discriminator = build_discriminator(input_dim=L)
    res_discriminator = build_res_discriminator(input_dim=L // 2)

    # 4. Trainer Initialization
    from src.trainer import EBNL_Trainer
    trainer = EBNL_Trainer(
        encoder=encoder,
        decoder_sys=decoder_sys,
        decoder_res=decoder_res,
        discriminator=discriminator,
        res_discriminator=res_discriminator,
        config={**config, "patience": current_patience},
        topology=topo,
    )
    parameter_stats = summarize_parameters(encoder, decoder_sys, decoder_res, discriminator, res_discriminator)
    
    # Training with Early Stopping
    train_stats = trainer.fit(train_ds, val_ds=val_ds, epochs=EP)
    
    # Save Weights
    save_path = os.path.join("weights", dataset_type, str(machine_id))
    os.makedirs(save_path, exist_ok=True)
    trainer.encoder.save_weights(f"{save_path}/encoder.weights.h5")
    trainer.decoder_sys.save_weights(f"{save_path}/decoder_sys.weights.h5")
    trainer.decoder_res.save_weights(f"{save_path}/decoder_res.weights.h5")
    # 5. Stream reconstruction errors directly into stitched branch scores.
    inference_start = time.perf_counter()
    scores_dir = os.path.join("scores", dataset_type, str(machine_id), "raw") if config.get("saved_score", False) else None
    test_stream = trainer.reconstruct_stitched(
        test_final,
        batch_size=TEST_BS,
        window_size=W,
        stride=test_stride,
        total_len=(len(test_final["res"]) - 1) * test_stride + W,
        window_indices=None,
        save_dir=scores_dir,
        split="test" if scores_dir else None,
    )
    inference_elapsed = time.perf_counter() - inference_start
    actual_len = test_stream["stitched_res"].shape[0] if test_stream["stitched_res"].shape[0] > 0 else test_stream["stitched_phy"].shape[0]
    inference_stats = summarize_inference(inference_elapsed, actual_len)

    def _compute_train_stream():
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

        train_stream = trainer.reconstruct_stitched(
            {"phy": train_phy, "res": train_res},
            batch_size=TEST_BS,
            window_size=W,
            stride=stride,
            total_len=train_total_len,
            window_indices=train_idx_shifted,
            save_dir=scores_dir,
            split="train" if scores_dir else None,
        )
        return train_stream, train_idx_shifted, train_total_len

    train_stream, train_idx_shifted, train_total_len = _compute_train_stream()
    if train_stream is None:
        raise RuntimeError("Train raw artifacts unavailable for external scoring.")

    test_labels = test_labels[:actual_len]
    
    print(f" Raw windows stitched to {actual_len} points | Labels: {len(test_labels)} | Anomaly Rate: {np.mean(test_labels):.2%}")
    if scores_dir:
        meta_path = os.path.join(scores_dir, "test_meta.npz")
        meta = dict(np.load(meta_path, allow_pickle=True))
        meta["labels"] = np.asarray(test_labels)
        np.savez_compressed(meta_path, **meta)

    stitched_scores = {
        "test_p": test_stream["stitched_phy"],
        "test_l": test_stream["stitched_res"],
        "train_p": train_stream["stitched_phy"],
        "train_l": train_stream["stitched_res"],
    }
    gc.collect()

    current_scored = score_stitched_entity(labels=test_labels, **stitched_scores)

    def _legacy_reduce(x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            return x
        if x.shape[1] == 0:
            return np.zeros(x.shape[0], dtype=np.float32)
        return np.mean(x, axis=1, dtype=np.float32)

    legacy_scores = {
        "test_p": _legacy_reduce(stitched_scores["test_p"]),
        "test_l": _legacy_reduce(stitched_scores["test_l"]),
        "train_p": _legacy_reduce(stitched_scores["train_p"]),
        "train_l": _legacy_reduce(stitched_scores["train_l"]),
    }
    legacy_scored = score_legacy_stitched_entity(labels=test_labels, **legacy_scores)

    scored = legacy_scored if config.get("legacy", False) else current_scored

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
    diagnosis_stats = None
    if config.get("enable_diagnosis", False):
        diagnosis_stats = compute_diagnosis_metrics(
            dataset=dataset_type,
            data_root=data_root,
            entity_id=str(machine_id),
            topology=topo,
            test_phy=stitched_scores["test_p"],
            test_res=stitched_scores["test_l"],
            train_phy=stitched_scores["train_p"],
            train_res=stitched_scores["train_l"],
            test_window_size=W,
            test_stride=test_stride,
            test_total_len=actual_len,
            train_window_size=W,
            train_stride=stride,
            train_total_len=train_total_len,
            test_window_indices=None,
            train_window_indices=train_idx_shifted,
        )
    if config.get("both", False):
        def _fmt(prefix, out):
            return (
                f"[{prefix}] AUC: {float(out['auc']):.4f} | PR-AUC: {float(out['prauc']):.4f} | "
                f"P: {float(out['p_best']):.4f} | R: {float(out['r_best']):.4f} | F1: {float(out['f1_best']):.4f} | "
                f"vusauc: {float(out['vusaucc']):.4f} | vuspr: {float(out['vuspr']):.4f} | "
                f"aff_p: {float(out['aff_p']):.4f} | aff_r: {float(out['aff_r']):.4f} | aff1: {float(out['aff1']):.4f}"
            )
        print(_fmt('RESULTS current', current_scored))
        print(_fmt('RESULTS legacy', legacy_scored))
    else:
        print(f"[RESULTS] AUC: {auc_score:.4f} | PR-AUC: {pr_auc_score:.4f} | P: {p_best:.4f} | R: {r_best:.4f} | F1: {f1_best:.4f} | vusauc: {vusauc:.4f} | vuspr: {vuspr:.4f} | aff_p: {aff_p:.4f} | aff_r: {aff_r:.4f} | aff1: {aff1:.4f}")
    print(
        f"[INFERENCE] Elapsed time: {inference_stats['elapsed_minutes']:.4f} mins | "
        f"Inference per sample: {inference_stats['ms_per_sample']:.4f} ms"
    )
    print(f"[PARAMS] Num. of params: {parameter_stats['total_params']}")
    K.clear_session()

    gc.collect()

    if config.get("both", False):
        return (
            current_scored, legacy_scored,
            train_stats, inference_stats, diagnosis_stats, parameter_stats,
            (encoder, decoder_sys, decoder_res, discriminator, res_discriminator)
        )

    return (
        auc_score, pr_auc_score, p_best, r_best, f1_best, vusauc, vuspr, aff_p, aff_r, aff1,
        train_stats, inference_stats, diagnosis_stats, parameter_stats,
        (encoder, decoder_sys, decoder_res, discriminator, res_discriminator)
    )
