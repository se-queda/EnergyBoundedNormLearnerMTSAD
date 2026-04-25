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


def score_legacy_stitched_entity(*, labels, test_p, test_l, train_p, train_l) -> dict[str, float]:
    labels = np.asarray(labels).astype(int)
    test_p = np.asarray(test_p, dtype=np.float32).reshape(-1)
    test_l = np.asarray(test_l, dtype=np.float32).reshape(-1)
    train_p = np.asarray(train_p, dtype=np.float32).reshape(-1)
    train_l = np.asarray(train_l, dtype=np.float32).reshape(-1)

    e_d = np.zeros(len(labels), dtype=np.float32)
    e_d_train = np.zeros(train_p.shape[0], dtype=np.float32)
    train_counts = np.ones(train_p.shape[0], dtype=np.float32)

    train_norm_stats = {
        'p': _robust_stats(train_p, train_counts),
        'l': _robust_stats(train_l, train_counts),
        'd': _robust_stats(e_d_train, train_counts),
    }

    r_p = _scale_with_stats(test_p, train_norm_stats['p'])
    r_l = _scale_with_stats(test_l, train_norm_stats['l'])
    _ = _scale_with_stats(e_d, train_norm_stats['d'])

    xor_pl = np.abs(r_p - r_l)
    bruteforce_max = np.maximum.reduce([r_p, r_l])
    point_scores = bruteforce_max + xor_pl
    point_scores = pd.Series(point_scores).ewm(alpha=0.08, adjust=False).mean().to_numpy()

    labels = labels[:len(point_scores)]
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


def score_legacy_entity(*, labels, recons, topo, phy_dim: int, test_stride: int, stride: int, W: int, actual_len: int, trainer, train_final, train_idx) -> dict[str, float]:
    raise RuntimeError('score_legacy_entity is deprecated; use score_legacy_stitched_entity with precomputed branch scores')
