import numpy as np


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def trailing_mean(scores, window=5):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores.copy()
    window = max(1, int(window))
    out = np.zeros_like(scores, dtype=float)
    running = 0.0
    # Causal moving average over the trailing window only.
    for t in range(len(scores)):
        running += scores[t]
        if t >= window:
            running -= scores[t - window]
        out[t] = running / min(t + 1, window)
    return out


def smoother(
    scores,
    short_window=4,
    long_window=16,
    raw_mix=0.35,
):

    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores.copy()

    raw_mix = _clip01(raw_mix)
    short_mean = trailing_mean(scores, short_window)
    long_mean = trailing_mean(scores, long_window)
    # Blend raw scores with short/long trailing consensus to smooth spikes.
    consensus = 0.5 * short_mean + 0.5 * long_mean
    return raw_mix * scores + (1.0 - raw_mix) * consensus
