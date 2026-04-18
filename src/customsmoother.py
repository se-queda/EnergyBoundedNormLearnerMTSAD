import numpy as np


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def ewma(scores, alpha=0.08):
    """Plain causal EWMA baseline."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores.copy()
    alpha = _clip01(alpha)
    out = np.zeros_like(scores, dtype=float)
    out[0] = scores[0]
    for t in range(1, len(scores)):
        out[t] = alpha * scores[t] + (1.0 - alpha) * out[t - 1]
    return out


def trailing_mean(scores, window=5):
    """Causal trailing mean over the last `window` points."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores.copy()
    window = max(1, int(window))
    out = np.zeros_like(scores, dtype=float)
    running = 0.0
    for t in range(len(scores)):
        running += scores[t]
        if t >= window:
            running -= scores[t - window]
        out[t] = running / min(t + 1, window)
    return out


def dual_timescale_accumulator(
    scores,
    alpha_fast=0.45,
    alpha_slow=0.08,
    mix=0.5,
):
    """
    Two-memory causal accumulator.

    The fast path reacts to sudden onsets. The slow path accumulates persistent
    support and mimics the stabilizing effect of retrospective window averaging.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores.copy()

    alpha_fast = _clip01(alpha_fast)
    alpha_slow = _clip01(alpha_slow)
    mix = _clip01(mix)

    fast = ewma(scores, alpha_fast)
    slow = ewma(scores, alpha_slow)
    return mix * fast + (1.0 - mix) * slow


def adaptive_evidence_smoother(
    scores,
    base_alpha=0.08,
    jerk_gain=0.35,
    variance_gain=0.20,
    hist_window=8,
    preserve_spikes=True,
):
    """
    Causal adaptive smoother intended to approximate non-causal window support.

    Intuition:
    - In calm regions, use a low alpha to accumulate evidence and suppress noise.
    - When score jerk or local variance rises, increase alpha so sharp anomaly
      onsets are not washed out.
    - Optionally preserve large raw spikes by taking the max with the raw score.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores.copy()

    base_alpha = _clip01(base_alpha)
    hist_window = max(2, int(hist_window))

    out = np.zeros_like(scores, dtype=float)
    out[0] = scores[0]

    for t in range(1, len(scores)):
        start = max(0, t - hist_window + 1)
        hist = scores[start : t + 1]

        local_var = float(np.var(hist)) if hist.size > 1 else 0.0
        local_scale = float(np.std(hist)) + 1e-8

        first_diff = scores[t] - scores[t - 1]
        prev_diff = scores[t - 1] - scores[t - 2] if t > 1 else 0.0
        jerk = abs(first_diff - prev_diff)

        jerk_term = jerk / local_scale
        var_term = local_var / (local_scale * local_scale + 1e-8)

        alpha_t = base_alpha + jerk_gain * np.tanh(jerk_term) + variance_gain * np.tanh(var_term)
        alpha_t = _clip01(alpha_t)

        smoothed = alpha_t * scores[t] + (1.0 - alpha_t) * out[t - 1]
        out[t] = max(smoothed, scores[t]) if preserve_spikes else smoothed

    return out


def causal_consensus_smoother(
    scores,
    short_window=4,
    long_window=16,
    raw_mix=0.35,
):
    """
    One-sided approximation to non-causal overlapping-window averaging.

    It combines:
    - raw score for immediate responsiveness
    - short trailing consensus
    - long trailing evidence accumulation
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores.copy()

    raw_mix = _clip01(raw_mix)
    short_mean = trailing_mean(scores, short_window)
    long_mean = trailing_mean(scores, long_window)
    consensus = 0.5 * short_mean + 0.5 * long_mean
    return raw_mix * scores + (1.0 - raw_mix) * consensus


SMOOTHERS = {
    "ewma": ewma,
    "dual_timescale": dual_timescale_accumulator,
    "adaptive_evidence": adaptive_evidence_smoother,
    "causal_consensus": causal_consensus_smoother,
}


def apply_smoother(scores, kind="adaptive_evidence", **kwargs):
    if kind not in SMOOTHERS:
        raise ValueError(f"unknown smoother '{kind}', available: {sorted(SMOOTHERS)}")
    return SMOOTHERS[kind](scores, **kwargs)
