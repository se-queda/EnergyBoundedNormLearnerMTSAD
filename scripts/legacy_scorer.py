from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.tsb_metrics import calculate_tsb_metrics as _tsb_metrics


METRIC_KEYS = ["auc", "prauc", "p_best", "r_best", "f1_best",
               "vusaucc", "vuspr", "aff_p", "aff_r", "aff1"]


def aggregate_scores(
    window_scores: np.ndarray,
    stride: int,
    window_size: int,
    total_len: int,
    window_indices: np.ndarray | None = None,
    return_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    x = np.asarray(window_scores, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected [N, W] tensor, got shape {x.shape}")

    scores = np.zeros(total_len, dtype=np.float32)
    counts = np.zeros(total_len, dtype=np.float32)
    prev_w_idx = None

    for i, window_val in enumerate(x):
        w_idx = int(window_indices[i]) if window_indices is not None else i
        start = w_idx * stride
        end = min(start + window_size, total_len)
        if end <= start:
            prev_w_idx = w_idx
            continue

        full_window = (prev_w_idx is None) or (w_idx - prev_w_idx > 1) or (stride >= window_size)
        slice_start = start if full_window else max(start, end - stride)
        if slice_start >= end:
            prev_w_idx = w_idx
            continue

        offset = slice_start - start
        slice_len = end - slice_start
        scores[slice_start:end] += window_val[offset: offset + slice_len]
        counts[slice_start:end] += 1
        prev_w_idx = w_idx

    scores = scores / np.maximum(counts, 1.0)
    if return_counts:
        return scores, counts
    return scores


def _robust_stats(s, counts=None):
    if counts is not None:
        mask = counts > 0
        if not np.any(mask):
            return 0.0, 0.0
        s = s[mask]
    if s.size == 0:
        return 0.0, 0.0
    med = float(np.nanmedian(s))
    q25 = float(np.nanpercentile(s, 25))
    q75 = float(np.nanpercentile(s, 75))
    iqr = q75 - q25
    return med, iqr


def _scale_with_stats(s, stats):
    med, iqr = stats
    if iqr < 1e-9:
        return np.zeros_like(s)
    return (s - med) / (iqr + 1e-6)


def score_legacy_entity(
    *,
    labels,
    recons,
    topo,
    phy_dim: int,
    test_stride: int,
    stride: int,
    W: int,
    actual_len: int,
    trainer,
    train_final,
    train_idx,
) -> dict[str, float]:
    if phy_dim > 0:
        se_p = np.mean(np.square(recons['phy_orig'] - recons['phy_hat']), axis=-1)
        e_p = aggregate_scores(se_p, test_stride, W, actual_len)
    else:
        e_p = np.zeros(actual_len)

    res_orig, res_hat = recons['res_orig'], recons['res_hat']
    if len(topo.res_to_lone_local) > 0:
        se_l = np.mean(
            np.square(res_orig[:, :, topo.res_to_lone_local] - res_hat[:, :, topo.res_to_lone_local]),
            axis=-1,
        )
        e_l = aggregate_scores(se_l, test_stride, W, actual_len)
    else:
        e_l = np.zeros(actual_len)

    e_d = np.zeros(actual_len)

    def _compute_train_scores():
        if train_idx is None or len(train_idx) == 0 or 'phy_anchor' not in train_final or 'res_orig' not in train_final:
            return None, None, None, None
        train_order = np.sort(train_idx)
        train_phy = train_final['phy_anchor'][train_order]
        train_res = train_final['res_orig'][train_order]
        train_idx_shifted = train_order
        num_train_windows = len(train_final['phy_anchor'])
        if len(train_idx_shifted) > 0:
            if train_idx_shifted.min() < 0 or train_idx_shifted.max() > num_train_windows - 1:
                raise RuntimeError(
                    f"Train window index out of bounds: expected within [0,{num_train_windows-1}] got [{train_idx_shifted.min()},{train_idx_shifted.max()}]."
                )
            train_total_len = (num_train_windows - 1) * stride + W
        else:
            train_total_len = 0

        if train_total_len <= 0:
            return None, None, None, None

        train_recons = trainer.reconstruct({'phy': train_phy, 'res': train_res})
        train_num_windows = train_recons['res_orig'].shape[0]
        dummy = np.ones((train_num_windows, W), dtype=float)
        _, train_counts = aggregate_scores(
            dummy,
            stride,
            W,
            train_total_len,
            window_indices=train_idx_shifted,
            return_counts=True,
        )

        if phy_dim > 0:
            se_p_train = np.mean(np.square(train_recons['phy_orig'] - train_recons['phy_hat']), axis=-1)
            e_p_train = aggregate_scores(se_p_train, stride, W, train_total_len, window_indices=train_idx_shifted)
        else:
            e_p_train = np.zeros(train_total_len)

        train_res_orig, train_res_hat = train_recons['res_orig'], train_recons['res_hat']
        if len(topo.res_to_lone_local) > 0:
            se_l_train = np.mean(
                np.square(
                    train_res_orig[:, :, topo.res_to_lone_local]
                    - train_res_hat[:, :, topo.res_to_lone_local]
                ),
                axis=-1,
            )
            e_l_train = aggregate_scores(se_l_train, stride, W, train_total_len, window_indices=train_idx_shifted)
        else:
            e_l_train = np.zeros(train_total_len)

        e_d_train = np.zeros(train_total_len)
        return e_p_train, e_l_train, e_d_train, train_counts

    e_p_train, e_l_train, e_d_train, train_counts = _compute_train_scores()
    if e_p_train is None:
        raise RuntimeError('Train stats unavailable for robust normalization.')

    train_norm_stats = {
        'p': _robust_stats(e_p_train, train_counts),
        'l': _robust_stats(e_l_train, train_counts),
        'd': _robust_stats(e_d_train, train_counts),
    }

    r_p = _scale_with_stats(e_p, train_norm_stats['p'])
    r_l = _scale_with_stats(e_l, train_norm_stats['l'])
    r_d = _scale_with_stats(e_d, train_norm_stats['d'])

    xor_pl = np.abs(r_p - r_l)
    restoration_signal = xor_pl
    bruteforce_max = np.maximum.reduce([r_p, r_l])
    point_scores = bruteforce_max + restoration_signal
    point_scores = pd.Series(point_scores).ewm(alpha=0.08, adjust=False).mean().to_numpy()

    labels = np.asarray(labels).astype(int)[:len(point_scores)]
    out = _tsb_metrics(point_scores, labels)
    return {
        'auc': float(out['auc']),
        'prauc': float(out['prauc']),
        'p_best': float(out['p_best']),
        'r_best': float(out['r_best']),
        'f1_best': float(out['f1_best']),
        'vusaucc': float(out['vusaucc']),
        'vuspr': float(out['vuspr']),
        'aff_p': float(out['aff_p']),
        'aff_r': float(out['aff_r']),
        'aff1': float(out['aff1']),
    }
