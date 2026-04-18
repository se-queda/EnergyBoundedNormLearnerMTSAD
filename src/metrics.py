import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import metrics
from tqdm import tqdm


def _segments_from_labels(labels):
    """
    Extract contiguous anomalous segments from binary labels.
    Returns a list of (start_idx, end_idx) inclusive.
    """
    segments = []
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == 1:
            start = i
            while i + 1 < n and labels[i + 1] == 1:
                i += 1
            end = i
            segments.append((start, end))
        i += 1
    return segments


def compute_auc(scores, labels):
    """Standard ROC-AUC on point-wise scores."""
    if np.all(labels == 0) or np.all(labels == 1):
        # AUC is undefined if only one class present; return 0.5 by convention
        return 0.5
    return roc_auc_score(labels, scores)


def compute_pr_auc(scores, labels):
    """PR-AUC (Average Precision) on point-wise scores."""
    if np.all(labels == 0) or np.all(labels == 1):
        return 0.0
    return average_precision_score(labels, scores)


# ----------------------------------------------------------------------
# Fc1: Composite F-score (Garg et al., 2021)
# ----------------------------------------------------------------------
def compute_fc1(preds, labels):
    """
    Composite F-score (Fc1) implementation.

    - Event-wise recall: fraction of ground-truth anomalous segments
      that intersect with at least one predicted positive point.
    - Time-wise precision: point-wise precision over all time steps.

    Fc1 = 2 * (event_recall * time_precision) / (event_recall + time_precision)
    """

    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    assert labels.shape == preds.shape

    # ---- Event-wise recall ----
    segments = _segments_from_labels(labels)
    if len(segments) == 0:
        # No anomalies in ground truth; Fc1 defined as 0 here
        return 0.0

    detected_segments = 0
    for (s, e) in segments:
        # If any prediction in this segment is 1, count segment as detected
        if np.any(preds[s:e + 1] == 1):
            detected_segments += 1

    event_recall = detected_segments / len(segments)

    # ---- Time-wise precision ----
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))

    if tp + fp == 0:
        time_precision = 0.0
    else:
        time_precision = tp / (tp + fp)

    if event_recall + time_precision == 0:
        return 0.0

    fc1 = 2 * event_recall * time_precision / (event_recall + time_precision)
    return fc1


# ----------------------------------------------------------------------
# PA%K (Kim et al., 2022)
# ----------------------------------------------------------------------
def pak(scores, targets, thres, k=20):
    """

    :param scores: anomaly scores
    :param targets: target labels
    :param thres: anomaly threshold
    :param k: PA%K ratio, 0 equals to conventional point adjust and 100 equals to original predictions
    :return: point_adjusted predictions
    """
    predicts = scores > thres
    actuals = targets > 0.01

    one_start_idx = np.where(np.diff(actuals, prepend=0) == 1)[0]
    zero_start_idx = np.where(np.diff(actuals, prepend=0) == -1)[0]

    assert len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(zero_start_idx)

    if len(one_start_idx) == len(zero_start_idx) + 1:
        zero_start_idx = np.append(zero_start_idx, len(predicts))

    for i in range(len(one_start_idx)):
        if predicts[one_start_idx[i]:zero_start_idx[i]].sum() > k / 100 * (zero_start_idx[i] - one_start_idx[i]):
            predicts[one_start_idx[i]:zero_start_idx[i]] = 1

    return predicts



def evaluate(scores, targets, pa=True, interval=10, k=0):
    """
    :param scores: list or np.array or tensor, anomaly score
    :param targets: list or np.array or tensor, target labels
    :param pa: True/False
    :param interval: threshold search interval
    :param k: PA%K threshold
    :return: results dictionary
    """
    assert len(scores) == len(targets)

    results = {}

    try:
        scores = np.asarray(scores)
        targets = np.asarray(targets)
    except TypeError:
        scores = np.asarray(scores.cpu())
        targets = np.asarray(targets.cpu())

    precision, recall, threshold = metrics.precision_recall_curve(targets, scores)
    f1_score = 2 * precision * recall / (precision + recall + 1e-12)

    results['best_f1_wo_pa'] = np.max(f1_score)
    results['best_precision_wo_pa'] = precision[np.argmax(f1_score)]
    results['best_recall_wo_pa'] = recall[np.argmax(f1_score)]
    results['prauc_wo_pa'] = metrics.average_precision_score(targets, scores)
    results['auc_wo_pa'] = metrics.roc_auc_score(targets, scores)

    if pa:
        # find F1 score with optimal threshold of best_f1_wo_pa
        pa_scores = pak(scores, targets, threshold[np.argmax(f1_score)], k)
        results['raw_f1_w_pa'] = metrics.f1_score(targets, pa_scores)
        results['raw_precision_w_pa'] = metrics.precision_score(targets, pa_scores)
        results['raw_recall_w_pa'] = metrics.recall_score(targets, pa_scores)

        # find best F1 score with varying thresholds
        if len(scores) // interval < 1:
            ths = threshold
        else:
            ths = [threshold[interval*i] for i in range(len(threshold)//interval)]
        pa_f1_scores = [metrics.f1_score(targets, pak(scores, targets, th, k)) for th in tqdm(ths)]
        pa_f1_scores = np.asarray(pa_f1_scores)
        results['best_f1_w_pa'] = np.max(pa_f1_scores)
        results['best_f1_th_w_pa'] = ths[np.argmax(pa_f1_scores)]
        pa_scores = pak(scores, targets, ths[np.argmax(pa_f1_scores)], k)
        results['best_precision_w_pa'] = metrics.precision_score(targets, pa_scores)
        results['best_recall_w_pa'] = metrics.recall_score(targets, pa_scores)
        results['pa_f1_scores'] = pa_f1_scores

    return results


import numpy as np

def early_trigger_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    diffs = np.diff(y_true, prepend=0)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    if len(ends) < len(starts):
        ends = np.append(ends, len(y_true))
        
    scores = []
    
    for s, e in zip(starts, ends):
        length = e - s

        segment_pred = y_pred[s:e]
        

        trigger_indices = np.where(segment_pred == 1)[0]
        
        if len(trigger_indices) == 0:

            scores.append(0.0)
        else:
  
            t_first = trigger_indices[0]
            score = 1.0 - (t_first / length)
            scores.append(score)
            
    if not scores:
        return 0.0
        
    return np.mean(scores)
