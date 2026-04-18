from __future__ import annotations


def summarize_inference(elapsed_seconds: float, num_samples: int) -> dict:
    """Return simple runtime metrics in the same style as SARAD."""
    elapsed_seconds = max(float(elapsed_seconds), 0.0)
    num_samples = max(int(num_samples), 1)
    return {
        "elapsed_minutes": elapsed_seconds / 60.0,
        "ms_per_sample": (elapsed_seconds * 1000.0) / num_samples,
    }
