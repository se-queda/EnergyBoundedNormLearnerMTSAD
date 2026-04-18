from __future__ import annotations


def summarize_parameters(*models) -> dict:
    total = 0
    trainable = 0
    non_trainable = 0

    for model in models:
        if model is None:
            continue
        for var in getattr(model, "variables", []):
            n = int(var.shape.num_elements() or 0)
            total += n
            if getattr(var, "trainable", False):
                trainable += n
            else:
                non_trainable += n

    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
    }
