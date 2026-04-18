import os
import random
import numpy as np
import tensorflow as tf


def initialize_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    print(f" Global seed set to: {seed}")


def _load_suite_seeds(path="results/suite_seeds.txt"):
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        seeds = [int(x.strip()) for x in f.read().split(",") if x.strip()]
    return seeds if seeds else None


def _next_suite_seed(seeds, index_path="results/suite_seed_index.txt"):
    if not seeds:
        return None
    idx = 0
    if os.path.isfile(index_path):
        try:
            idx = int(open(index_path, "r").read().strip())
        except Exception:
            idx = 0
    seed = seeds[idx % len(seeds)]
    try:
        with open(index_path, "w") as f:
            f.write(str((idx + 1) % len(seeds)))
    except Exception:
        pass
    return seed
suite_seeds = _load_suite_seeds()
env_seed = os.environ.get("SUITE_SEED")
if env_seed is not None:
    try:
        initialize_seeds(int(env_seed))
    except Exception:
        initialize_seeds()
elif suite_seeds:
    seed = _next_suite_seed(suite_seeds)
    initialize_seeds(seed if seed is not None else 42)
else:
    initialize_seeds()
