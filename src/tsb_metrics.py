import numpy as np
import math
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from TSB_UAD.vus.metrics import get_metrics

def get_list_anomaly(labels):
    results = []
    start = 0
    anom = False
    for i, val in enumerate(labels):
        if val == 1:
            anom = True
        else:
            if anom:
                results.append(i - start)
                anom = False
        if not anom:
            start = i
    if anom:
        results.append(len(labels) - start)
    return results

def get_median_anomaly_length(labels):
    lengths = get_list_anomaly(labels)
    if not lengths:
        return 100
    return int(np.median(lengths))

def calculate_tsb_metrics(scores, labels):
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    valid_len = min(len(scores), len(labels))
    scores = scores[:valid_len]
    labels = labels[:valid_len]

    if valid_len == 0:
        raise ValueError("scores and labels must be non-empty")
    if np.unique(labels).size < 2:
        raise ValueError("labels must contain both normal and anomalous classes")
    auc = float(roc_auc_score(labels, scores))
    prauc = float(average_precision_score(labels, scores))
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-12)
    best_idx = np.argmax(f1_curve)
    p_best, r_best, f1_best = float(precision_curve[best_idx]), float(recall_curve[best_idx]), float(f1_curve[best_idx])
    median_len = get_median_anomaly_length(labels)
    sliding_window = 2 * median_len
    
    tsb_out = get_metrics(scores, labels, metric="all", slidingWindow=sliding_window)
    v_roc = tsb_out.get("VUS_ROC") or tsb_out.get("vus-roc") or tsb_out.get("vusaucc") or 0.0
    v_pr  = tsb_out.get("VUS_PR") or tsb_out.get("vus-pr") or tsb_out.get("vuspr") or 0.0
    
    aff_p = tsb_out.get("Affiliation_Precision") or tsb_out.get("aff-p") or 0.0
    aff_r = tsb_out.get("Affiliation_Recall") or tsb_out.get("aff-r") or 0.0
    
    v_roc = float(v_roc) if not math.isnan(float(v_roc)) else 0.0
    v_pr  = float(v_pr) if not math.isnan(float(v_pr)) else 0.0
    aff_p = float(aff_p) if not math.isnan(float(aff_p)) else 0.0
    aff_r = float(aff_r) if not math.isnan(float(aff_r)) else 0.0
    
    aff1 = 0.0 if (aff_p + aff_r) == 0 else (2 * aff_p * aff_r / (aff_p + aff_r))

    return {
        "auc": auc,
        "prauc": prauc,
        "p_best": p_best,
        "r_best": r_best,
        "f1_best": f1_best,
        "vusaucc": v_roc,
        "vuspr": v_pr,
        "aff_p": aff_p,
        "aff_r": aff_r,
        "aff1": aff1,
        "sliding_window": sliding_window
    }
