from __future__ import annotations

import math
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd


DIAG_KEYS = ["hr_100", "hr_150", "ndcg_100", "ndcg_150", "ips_100", "ips_150"]


def _stitch_feature_scores(
    x: np.ndarray | None,
    stride: int,
    window_size: int,
    total_len: int,
    window_indices: np.ndarray | None = None,
) -> np.ndarray | None:
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected [N, W, F] tensor, got shape {x.shape}")
    if x.shape[-1] == 0:
        return np.zeros((total_len, 0), dtype=np.float32)

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


def _scale_identity(x: np.ndarray | None) -> np.ndarray | None:
    if x is None:
        return None
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(x, 0.0)


def _calibrate_standard(test: np.ndarray | None, train: np.ndarray | None) -> np.ndarray | None:
    if test is None:
        return None
    test = np.asarray(test, dtype=np.float32)
    if train is None or np.asarray(train).size == 0:
        return test
    train = np.asarray(train, dtype=np.float32)
    mu = np.mean(train, axis=0)
    sigma = np.std(train, axis=0)
    return (test - mu) / np.maximum(sigma, 1e-6)


def _dcg_at_k(relevances: list[int], k: int) -> float:
    out = 0.0
    for idx, rel in enumerate(relevances[:k], start=1):
        out += rel / math.log2(idx + 1)
    return out


def _compute_rank_metrics(intervals: Iterable[tuple[int, int, list[int]]], diagno_scores: np.ndarray) -> dict[str, float]:
    hr_scores = {100: [], 150: []}
    ndcg_scores = {100: [], 150: []}
    ips_scores = {100: [], 150: []}

    for start_idx, end_idx, gt_features in intervals:
        gt = sorted(set(int(v) for v in gt_features if int(v) >= 0))
        if not gt:
            continue
        start_idx = max(0, int(start_idx))
        end_idx = min(diagno_scores.shape[0] - 1, int(end_idx))
        if end_idx < start_idx:
            continue

        for t in range(start_idx, end_idx + 1):
            feature_scores = diagno_scores[t]
            ranking = np.argsort(-feature_scores)
            for p in (100, 150):
                k = int(math.ceil(len(gt) * p / 100.0))
                topk = ranking[:k].tolist()
                overlap = len(set(gt).intersection(topk))
                hr_scores[p].append(overlap / len(gt))
                relevances = [1 if feat in gt else 0 for feat in topk]
                idcg = sum(1.0 / math.log2(i + 1) for i in range(2, len(gt) + 2))
                ndcg_scores[p].append(_dcg_at_k(relevances, k) / idcg if idcg > 0 else 0.0)

        range_scores = np.max(diagno_scores[start_idx:end_idx + 1], axis=0)
        range_ranking = np.argsort(-range_scores)
        for p in (100, 150):
            k = int(math.ceil(len(gt) * p / 100.0))
            topk = range_ranking[:k].tolist()
            overlap = len(set(gt).intersection(topk))
            ips_scores[p].append(overlap / len(gt))

    return {
        "hr_100": float(np.mean(hr_scores[100])) if hr_scores[100] else math.nan,
        "hr_150": float(np.mean(hr_scores[150])) if hr_scores[150] else math.nan,
        "ndcg_100": float(np.mean(ndcg_scores[100])) if ndcg_scores[100] else math.nan,
        "ndcg_150": float(np.mean(ndcg_scores[150])) if ndcg_scores[150] else math.nan,
        "ips_100": float(np.mean(ips_scores[100])) if ips_scores[100] else math.nan,
        "ips_150": float(np.mean(ips_scores[150])) if ips_scores[150] else math.nan,
    }


def _parse_smd_intervals(label_path: str) -> list[tuple[int, int, list[int]]]:
    intervals = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            range_part, numbers_part = line.split(":")
            start, end = map(int, range_part.split("-"))
            numbers = [int(num) - 1 for num in numbers_part.split(",") if num.strip()]
            intervals.append((start, end - 1, numbers))
    return intervals


def _infer_time_column(df: pd.DataFrame) -> str | None:
    first_col = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[first_col]):
        return first_col
    for col in df.columns:
        name = str(col).strip().lower()
        if "timestamp" in name or name == "time":
            return col
    return None


def _normalize_feature_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _parse_swat_intervals(data_root: str, feature_cols: list[str]) -> list[tuple[int, int, list[int]]]:
    test_path = os.path.join(data_root, "Test.xlsx")
    attack_path = os.path.join(data_root, "List_of_attacks_Final.xlsx")
    test_df = pd.read_excel(test_path).ffill()
    if all(str(c).strip().lower().startswith("unnamed") for c in test_df.columns):
        test_df = pd.read_excel(test_path, header=1).ffill()
    attack_df = pd.read_excel(attack_path)

    time_col = _infer_time_column(test_df)
    if time_col is None:
        return []
    test_times = pd.to_datetime(test_df[time_col], errors="coerce")
    if test_times.isna().all():
        return []

    feature_map = {_normalize_feature_name(col): idx for idx, col in enumerate(feature_cols)}
    intervals = []

    for _, row in attack_df.iterrows():
        attack_point = row.get("Attack Point")
        start_val = row.get("Start Time")
        end_val = row.get("End Time")
        if pd.isna(attack_point) or pd.isna(start_val) or pd.isna(end_val):
            continue

        point_text = str(attack_point)
        if "no physical impact" in point_text.strip().lower():
            continue

        tokens = re.findall(r"[A-Za-z]+-\d+", point_text)
        gt = []
        for token in tokens:
            idx = feature_map.get(_normalize_feature_name(token))
            if idx is not None:
                gt.append(idx)
        if not gt:
            continue

        start_ts = pd.to_datetime(start_val, errors="coerce")
        end_ts = pd.to_datetime(end_val, errors="coerce")
        if pd.isna(start_ts) or pd.isna(end_ts):
            continue

        mask = (test_times >= start_ts) & (test_times <= end_ts)
        idxs = np.flatnonzero(mask.to_numpy())
        if idxs.size == 0:
            continue
        intervals.append((int(idxs[0]), int(idxs[-1]), gt))

    return intervals


def compute_diagnosis_metrics(
    *,
    dataset: str,
    data_root: str,
    entity_id: str,
    topology,
    test_phy,
    test_res,
    train_phy,
    train_res,
    test_window_size: int,
    test_stride: int,
    test_total_len: int,
    train_window_size: int,
    train_stride: int,
    train_total_len: int,
    test_window_indices=None,
    train_window_indices=None,
) -> dict[str, float] | None:
    dataset = dataset.upper()
    if dataset not in {"SMD", "SWAT"}:
        return None
    if dataset == "SMD" and str(entity_id) == "SMD_Compact":
        return None

    test_p = _stitch_feature_scores(test_phy, test_stride, test_window_size, test_total_len, test_window_indices)
    test_r = _stitch_feature_scores(test_res, test_stride, test_window_size, test_total_len, test_window_indices)
    train_p = _stitch_feature_scores(train_phy, train_stride, train_window_size, train_total_len, train_window_indices)
    train_r = _stitch_feature_scores(train_res, train_stride, train_window_size, train_total_len, train_window_indices)

    test_p = _calibrate_standard(_scale_identity(test_p), _scale_identity(train_p))
    test_r = _calibrate_standard(_scale_identity(test_r), _scale_identity(train_r))

    total_sensors = len(topology.idx_phy) + len(topology.idx_res)
    full_scores = np.zeros((test_total_len, total_sensors), dtype=np.float32)
    if test_p is not None and test_p.shape[1] > 0:
        full_scores[:, np.asarray(topology.idx_phy, dtype=int)] = test_p
    if test_r is not None and test_r.shape[1] > 0:
        full_scores[:, np.asarray(topology.idx_res, dtype=int)] = test_r

    if dataset == "SMD":
        label_path = os.path.join(data_root, "interpretation_label", f"{entity_id}.txt")
        if not os.path.isfile(label_path):
            return None
        intervals = _parse_smd_intervals(label_path)
    else:
        feature_cols = [f"S{i+1}" for i in range(total_sensors)]
        # SWAT uses real tag names; rebuild exact feature-column order.
        train_path = os.path.join(data_root, "Train.xlsx")
        test_path = os.path.join(data_root, "Test.xlsx")
        train_df = pd.read_excel(train_path).ffill().bfill()
        test_df = pd.read_excel(test_path).ffill()
        if all(str(c).strip().lower().startswith("unnamed") for c in train_df.columns):
            train_df = pd.read_excel(train_path, header=1).ffill().bfill()
        if all(str(c).strip().lower().startswith("unnamed") for c in test_df.columns):
            test_df = pd.read_excel(test_path, header=1).ffill()
        train_df.columns = [str(c).strip() for c in train_df.columns]
        test_df.columns = [str(c).strip() for c in test_df.columns]
        label_col = next((c for c in test_df.columns if str(c).strip().lower() == "normal/attack"), None)
        if label_col and label_col in test_df.columns:
            test_df = test_df.drop(columns=[label_col])
        if label_col and label_col in train_df.columns:
            train_df = train_df.drop(columns=[label_col])
        for df in (train_df, test_df):
            time_col = _infer_time_column(df)
            if time_col in df.columns:
                df.drop(columns=[time_col], inplace=True)
        common_cols = [c for c in train_df.columns if c in test_df.columns]
        feature_cols = common_cols
        intervals = _parse_swat_intervals(data_root, feature_cols)

    if not intervals:
        return None
    return _compute_rank_metrics(intervals, full_scores)
