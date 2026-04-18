import tensorflow as tf
import numpy as np



def mix_features(z_orig, z_other): #latent mixing
    batch_size = tf.shape(z_orig)[0]
    alpha = tf.random.uniform((batch_size, 1), minval=0.0, maxval=1.0)
    z_mixed = alpha * z_orig + (1.0 - alpha) * z_other
    return z_mixed, alpha

def random_masker(data_windows, mask_rates=(0.05, 0.15, 0.30, 0.50), seed=None):#masker
    rng = np.random.default_rng(seed) if seed is not None else None
    views = []
    N, T, F = data_windows.shape
    for rate in mask_rates:
        random_values = rng.random((N, F)) if rng is not None else np.random.random((N, F))
        mask = random_values < rate
        view = data_windows.copy()
        view *= (~mask[:, None, :])
        views.append(view)
    return tuple(views)
