#!/usr/bin/env python3
"""Standalone scorer that reads raw reconstruction artifacts and writes metrics."""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.smoother import smoother
from src.tsb_metrics import calculate_tsb_metrics as _tsb_metrics

SCALING     = "identity"
MIXING      = "sum"
SMOOTHER    = "causal_consensus"
CALIB       = "standard"
PLACEMENT   = "sensor_post"
GRANULARITY = "branch"

DEFAULT_OUT_ROOT = REPO_ROOT / "posthoc_runs" / "generalized_anomaly_scorer"

METRIC_KEYS = ["auc", "prauc", "p_best", "r_best", "f1_best",
               "vusaucc", "vuspr", "aff_p", "aff_r", "aff1"]

ENTITY_HEADER = (
    ["dataset", "scaling_method", "mixing_method", "ewma_alpha", "smoother",
     "train_calibration", "placement", "mix_granularity", "id"]
    + METRIC_KEYS
)
SUMMARY_HEADER = (
    ["dataset", "scaling_method", "mixing_method", "ewma_alpha", "smoother",
     "train_calibration", "placement", "mix_granularity"]
    + METRIC_KEYS
)

def _meta_int(meta, key: str) -> int:
    if key not in meta:
        raise KeyError(
            f"Key '{key}' not found in meta. "
            "This indicates a corrupted metadata file. Re-run training."
        )
    return int(np.asarray(meta[key]).item())


def _meta_window_indices(meta) -> np.ndarray | None:
    if "window_indices" not in meta:
        return None
    arr = np.asarray(meta["window_indices"])
    if arr.shape == ():
        val = arr.item()
        if val is None:
            return None
        return np.asarray(val, dtype=np.int64)
    return np.asarray(arr, dtype=np.int64)


def _load_raw_array(base_path: Path) -> np.ndarray | None:
    npz_path = base_path.with_suffix(".npz")
    if npz_path.exists():
        z = np.load(npz_path, allow_pickle=True)
        return np.asarray(z["err"], dtype=np.float32)
    return None


def load_entity(entity_dir: Path) -> dict[str, Any]:
    """Load test/train raw errors and their window metadata."""
    raw_dir = entity_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw dir: {raw_dir}")

    test_meta = np.load(raw_dir / "test_meta.npz", allow_pickle=True)
    train_meta_path = raw_dir / "train_meta.npz"
    if not train_meta_path.exists():
        raise FileNotFoundError(f"Missing train metadata file: {train_meta_path}")
    train_meta = np.load(train_meta_path, allow_pickle=True)

    test_window_size = _meta_int(test_meta, "window_size")
    test_stride = _meta_int(test_meta, "stride")
    test_total_len = _meta_int(test_meta, "total_len")
    test_win_idx = _meta_window_indices(test_meta)

    labels = np.asarray(test_meta["labels"]).astype(int)

    test_phy = _load_raw_array(raw_dir / "test_phy_raw")
    test_res = _load_raw_array(raw_dir / "test_res_raw")
    train_phy = _load_raw_array(raw_dir / "train_phy_raw")
    train_res = _load_raw_array(raw_dir / "train_res_raw")

    test_lengths = [n for n in (
        test_phy.shape[0] if test_phy is not None else None,
        test_res.shape[0] if test_res is not None else None,
    ) if n is not None]
    if not test_lengths:
        raise FileNotFoundError(f"No test_phy_raw or test_res_raw found in {raw_dir}")
    if len(set(test_lengths)) > 1:
        raise ValueError(
            f"Mismatched test raw window counts in {raw_dir}: {test_lengths}"
        )

    train_window_size = _meta_int(train_meta, "window_size")
    train_stride = _meta_int(train_meta, "stride")
    train_total_len = _meta_int(train_meta, "total_len")
    train_win_idx = _meta_window_indices(train_meta)

    train_lengths = [n for n in (
        train_phy.shape[0] if train_phy is not None else None,
        train_res.shape[0] if train_res is not None else None,
    ) if n is not None]
    if train_lengths and len(set(train_lengths)) > 1:
        raise ValueError(
            f"Mismatched train raw window counts in {raw_dir}: {train_lengths}"
        )

    return {
        "labels":            labels,
        "test_phy":          test_phy,
        "test_res":          test_res,
        "train_phy":         train_phy,
        "train_res":         train_res,
        "test_window_size":  test_window_size,
        "test_stride":       test_stride,
        "test_total_len":    test_total_len,
        "test_window_indices": test_win_idx,
        "train_window_size": train_window_size,
        "train_stride":      train_stride,
        "train_total_len":   train_total_len,
        "train_window_indices": train_win_idx,
    }


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
    dataset: str,
    entity_id: str,
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
    out_dir: str | Path | None = None,
) -> dict[str, float]:
    """Score one entity directly from in-memory raw artifacts."""
    dataset = str(dataset).upper()
    entity_id = str(entity_id)
    if out_dir is None:
        out_dir = DEFAULT_OUT_ROOT / dataset / entity_id

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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
    m = score_entity(entity_data)

    cfg_fields = [SCALING, MIXING, "none", SMOOTHER, CALIB, PLACEMENT, GRANULARITY]
    _write_csv(
        out_dir / f"{dataset.lower()}_entity_metrics.csv",
        ENTITY_HEADER,
        [[dataset] + cfg_fields + [entity_id] + [m[k] for k in METRIC_KEYS]],
    )
    return m


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


def _write_csv(path: Path, header: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerows(rows)


def score_dataset(
    dataset: str,
    scores_root: str | Path,
    out_dir: str | Path = DEFAULT_OUT_ROOT,
    entities: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Score all entities in one dataset and write entity/avg CSVs."""
    dataset = str(dataset).upper()
    dataset_dir = Path(scores_root) / dataset
    entity_dirs = sorted([p for p in dataset_dir.iterdir()
                          if p.is_dir() and (p / "raw").exists()])
    if entities:
        allowed = set(map(str, entities))
        entity_dirs = [p for p in entity_dirs if p.name in allowed]
    if not entity_dirs:
        raise FileNotFoundError(f"No entity raw dirs found under {dataset_dir}")

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, float]] = {}
    entity_rows = []

    cfg_fields = [SCALING, MIXING, "none",
                  SMOOTHER, CALIB,
                  PLACEMENT, GRANULARITY]

    for entity_dir in entity_dirs:
        eid = entity_dir.name
        entity_data = load_entity(entity_dir)
        m = score_entity(entity_data)
        all_results[eid] = m
        entity_rows.append(
            [dataset] + cfg_fields + [eid] + [m[k] for k in METRIC_KEYS]
        )

    _write_csv(out_dir / f"{dataset.lower()}_entity_metrics.csv",
               ENTITY_HEADER, entity_rows)

    # Mean across entity-level metric rows.
    means = []
    for k in METRIC_KEYS:
        vals = [r[k] for r in all_results.values()]
        means.append(float(np.mean(vals)))
    _write_csv(out_dir / f"{dataset.lower()}_avg_metrics.csv",
               SUMMARY_HEADER, [[dataset] + cfg_fields + means])

    return all_results


def score_single_entity(
    dataset: str,
    entity_id: str,
    scores_root: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, float]:
    """Score a single entity from its raw artifact directory."""
    dataset   = str(dataset).upper()
    entity_id = str(entity_id)
    if out_dir is None:
        out_dir = DEFAULT_OUT_ROOT / dataset / entity_id

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    entity_dir = Path(scores_root) / dataset / entity_id
    if not entity_dir.exists():
        raise FileNotFoundError(f"Score artifact directory not found: {entity_dir}")

    entity_data = load_entity(entity_dir)
    m = score_entity(entity_data)

    cfg_fields = [SCALING, MIXING, "none",
                  SMOOTHER, CALIB,
                  PLACEMENT, GRANULARITY]
    _write_csv(
        out_dir / f"{dataset.lower()}_entity_metrics.csv",
        ENTITY_HEADER,
        [[dataset] + cfg_fields + [entity_id] + [m[k] for k in METRIC_KEYS]],
    )

    return m


def run_fixed_generalizable_sweep(
    datasets,
    scores_root: str | Path,
    out_dir: str | Path = DEFAULT_OUT_ROOT,
    entities=None,
) -> Path:
    """Score all requested datasets into one output root."""
    out_dir = Path(out_dir).resolve()
    for dataset in datasets:
        score_dataset(dataset, scores_root, out_dir=out_dir, entities=entities)
    return out_dir


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            f"Score anomaly detections from raw reconstruction artifacts: "
            f"{SCALING}|{MIXING}|{SMOOTHER}|"
            f"{CALIB}|{PLACEMENT}|{GRANULARITY}"
        )
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--entities", nargs="*", default=None)
    parser.add_argument("--scores-root", required=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    results = run_fixed_generalizable_sweep(
        datasets=args.datasets,
        scores_root=args.scores_root,
        out_dir=args.out_dir,
        entities=args.entities,
    )

    print(f"\nResults written to: {results}")
    print(f"Scorer config: {SCALING}|{MIXING}|"
          f"{SMOOTHER}|{CALIB}|{PLACEMENT}|"
          f"{GRANULARITY}")


if __name__ == "__main__":
    main()
