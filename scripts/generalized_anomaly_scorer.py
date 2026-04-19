
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.smoother import smoother
from src.tsb_metrics import calculate_tsb_metrics as _tsb_metrics

METRIC_KEYS = ["auc", "prauc", "p_best", "r_best", "f1_best",
               "vusaucc", "vuspr", "aff_p", "aff_r", "aff1"]


def entity_data_from_raw(
    *,
    labels,
    test_phy,
    test_res,
    train_phy,
    train_res,
    test_window_size,
    test_stride,
    test_total_len,
    train_window_size,
    train_stride,
    train_total_len,
    test_window_indices=None,
    train_window_indices=None,
) -> dict[str, Any]:
    """Build an in-memory entity bundle with the same schema as load_entity()."""
    return {
        "labels": np.asarray(labels).astype(int),
        "test_phy": None if test_phy is None else np.asarray(test_phy, dtype=np.float32),
        "test_res": None if test_res is None else np.asarray(test_res, dtype=np.float32),
        "train_phy": None if train_phy is None else np.asarray(train_phy, dtype=np.float32),
        "train_res": None if train_res is None else np.asarray(train_res, dtype=np.float32),
        "test_window_size": int(test_window_size),
        "test_stride": int(test_stride),
        "test_total_len": int(test_total_len),
        "test_window_indices": None if test_window_indices is None else np.asarray(test_window_indices, dtype=np.int64),
        "train_window_size": int(train_window_size),
        "train_stride": int(train_stride),
        "train_total_len": int(train_total_len),
        "train_window_indices": None if train_window_indices is None else np.asarray(train_window_indices, dtype=np.int64),
    }


def score_raw_entity(
    *,
    labels,
    test_phy,
    test_res,
    train_phy,
    train_res,
    test_window_size,
    test_stride,
    test_total_len,
    train_window_size,
    train_stride,
    train_total_len,
    test_window_indices=None,
    train_window_indices=None,
) -> dict[str, float]:
    """Score one entity directly from in-memory raw artifacts."""
    entity_data = entity_data_from_raw(
        labels=labels,
        test_phy=test_phy,
        test_res=test_res,
        train_phy=train_phy,
        train_res=train_res,
        test_window_size=test_window_size,
        test_stride=test_stride,
        test_total_len=test_total_len,
        train_window_size=train_window_size,
        train_stride=train_stride,
        train_total_len=train_total_len,
        test_window_indices=test_window_indices,
        train_window_indices=train_window_indices,
    )
    return score_entity(entity_data)


def _stitch_feature_scores(
    x: np.ndarray | None,
    stride: int,
    window_size: int,
    total_len: int,
    window_indices: np.ndarray | None = None,
) -> np.ndarray | None:
    """Map raw error windows [N, W, F] to stitched point scores [T, F]."""
    if x is None:
        return None
    if x.ndim == 2:
        raise ValueError(
            f"Expected raw window tensor [N, W, F], received legacy point tensor {x.shape}"
        )
    if x.ndim != 3:
        raise ValueError(f"Expected [N, W, F] tensor, got shape {x.shape}")
    if x.shape[-1] == 0:
        return np.zeros((total_len, 0), dtype=np.float32)

    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0)
    scores = np.zeros((total_len, x.shape[-1]), dtype=np.float32)
    counts = np.zeros(total_len, dtype=np.float32)
    prev_w_idx = None

    stitch_bar = tqdm(
        enumerate(x),
        total=len(x),
        desc="Stitch windows",
        unit="window",
    )
    for i, window_val in stitch_bar:
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
        stitch_bar.set_postfix({"done": f"{min(i + 1, len(x))}/{len(x)}"})

    return scores / np.maximum(counts[:, None], 1.0)


def _scale_identity(x: np.ndarray) -> np.ndarray:
    """Apply non-negative identity scaling elementwise."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(x, 0.0)


def _mix_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add two branch score vectors after length alignment."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = min(a.shape[0], b.shape[0])
    return a[:n] + b[:n]


def _calibrate_standard(test: np.ndarray, train: np.ndarray | None) -> np.ndarray:
    """Apply featurewise z-score calibration using train statistics."""
    test = np.asarray(test, dtype=np.float32)
    if train is None or (isinstance(train, np.ndarray) and train.size == 0):
        return test
    train = np.asarray(train, dtype=np.float32)
    axis = 0 if train.ndim > 1 else None
    mu    = np.mean(train, axis=axis)
    sigma = np.std(train, axis=axis)
    return (test - mu) / np.maximum(sigma, 1e-6)


def _reduce_branch(x: np.ndarray) -> np.ndarray:
    """Reduce stitched feature scores [T, F] to a branch score [T]."""
    if x is None:
        return np.array([], dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=np.float32)
    return np.mean(x, axis=1, dtype=np.float32)


def _apply_score_smoother(x: np.ndarray) -> np.ndarray:
    """Apply the configured one-dimensional smoother."""
    return smoother(x)


def score_entity(entity_data: dict[str, Any]) -> dict[str, float]:
    """Run the fixed scorer on one entity."""
    W  = int(entity_data["test_window_size"])
    S  = int(entity_data["test_stride"])
    TL = int(entity_data["test_total_len"])
    labels = np.asarray(entity_data["labels"]).astype(int)[:TL]

    tw_idx = entity_data.get("test_window_indices")
    rw_idx = entity_data.get("train_window_indices")
    train_W = int(entity_data["train_window_size"])
    train_S = int(entity_data["train_stride"])
    train_TL = int(entity_data["train_total_len"])

    # 1. Stitch raw test/train windows to pointwise feature scores [T, F].
    test_p  = _stitch_feature_scores(entity_data["test_phy"],   S, W, TL, tw_idx)
    test_l  = _stitch_feature_scores(entity_data["test_res"],   S, W, TL, tw_idx)
    train_p = _stitch_feature_scores(entity_data["train_phy"], train_S, train_W, train_TL, rw_idx)
    train_l = _stitch_feature_scores(entity_data["train_res"], train_S, train_W, train_TL, rw_idx)

    if test_p is None:
        test_p = np.zeros((TL, 0), dtype=np.float32)
    if test_l is None:
        test_l = np.zeros((TL, 0), dtype=np.float32)
    if train_p is None:
        train_p = np.zeros((0, test_p.shape[1] or 1), dtype=np.float32)
    if train_l is None:
        train_l = np.zeros((0, test_l.shape[1] or 1), dtype=np.float32)

    # 2. Scale pointwise branch features.
    test_p_scaled  = _scale_identity(test_p)
    test_l_scaled  = _scale_identity(test_l)
    train_p_scaled = _scale_identity(train_p)
    train_l_scaled = _scale_identity(train_l)

    # 3. Calibrate test features with train statistics.
    test_p_cal = _calibrate_standard(test_p_scaled, train_p_scaled)
    test_l_cal = _calibrate_standard(test_l_scaled, train_l_scaled)

    # 4. Reduce each branch from [T, F] to [T].
    branch_p = _reduce_branch(test_p_cal)
    branch_l = _reduce_branch(test_l_cal)

    # 5. Mix the branch score streams.
    point_scores = _mix_sum(branch_p, branch_l)

    # 6. Smooth the final point score stream.
    point_scores = _apply_score_smoother(point_scores)

    if len(point_scores) != len(labels):
        raise ValueError(
            f"Score/label length mismatch: scores={len(point_scores)} labels={len(labels)}"
        )

    return _compute_metrics(point_scores, labels)

def _compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if labels.sum() <= 0:
        raise ValueError("labels must contain at least one anomalous point")
    out = _tsb_metrics(scores, labels)
    return {
        "auc":     float(out["auc"]),
        "prauc":   float(out["prauc"]),
        "p_best":  float(out["p_best"]),
        "r_best":  float(out["r_best"]),
        "f1_best": float(out["f1_best"]),
        "vusaucc": float(out["vusaucc"]),
        "vuspr":   float(out["vuspr"]),
        "aff_p":   float(out["aff_p"]),
        "aff_r":   float(out["aff_r"]),
        "aff1":    float(out["aff1"]),
    }

