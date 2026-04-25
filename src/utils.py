import os
import numpy as np
import tensorflow as tf



def load_txt_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return np.loadtxt(path, delimiter=',', dtype=np.float32)
    except:
        return np.loadtxt(path, dtype=np.float32)



def build_tf_datasets(
    train_final,
    val_split=0.2,
    batch_size=128,
    val_normal_only=False,
    window_size=None,
    stride=None,
):

    phy_data = train_final['phy_views']   
    res_data = train_final['res_views'] 
    num_samples = len(phy_data)

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    if val_split <= 0 or num_samples == 0:
        train_idx = indices
        val_idx = indices[:0]
    else:
        if val_normal_only:
            train_labels = train_final.get("train_labels")
            window_labels = None
            if train_labels is not None:
                train_labels = np.asarray(train_labels).astype(int).flatten()
                if train_labels.shape[0] == num_samples:
                    window_labels = train_labels
                elif window_size is not None and stride is not None:
                    total_len = (num_samples - 1) * int(stride) + int(window_size)
                    if train_labels.shape[0] >= total_len:
                        window_labels = np.zeros(num_samples, dtype=int)
                        for i in range(num_samples):
                            start = i * int(stride)
                            end = start + int(window_size)
                            window_labels[i] = int(np.max(train_labels[start:end]))
            if window_labels is None:
                window_labels = np.zeros(num_samples, dtype=int)

            normal_idx = indices[window_labels[indices] == 0]
            if len(normal_idx) == 0:
                normal_idx = indices
            val_size = int(len(normal_idx) * val_split)
            if val_size <= 0 and val_split > 0:
                val_size = 1
            np.random.shuffle(normal_idx)
            val_idx = normal_idx[:val_size]
            train_idx = np.setdiff1d(indices, val_idx, assume_unique=False)
        else:
            num_train = int(num_samples * (1 - val_split))
            train_idx = indices[:num_train]
            val_idx = indices[num_train:]
    
    train_phy, train_res = phy_data[train_idx], res_data[train_idx]
    val_phy, val_res = phy_data[val_idx], res_data[val_idx]
    
    AUTOTUNE = tf.data.AUTOTUNE

    def make_dataset(phy, res, shuffle=False):
        phy = np.asarray(phy, dtype=np.float32)
        res = np.asarray(res, dtype=np.float32)

        output_signature = (
            tf.TensorSpec(shape=phy.shape[1:], dtype=tf.float32),
            tf.TensorSpec(shape=res.shape[1:], dtype=tf.float32),
        )

        def gen():
            for i in range(len(phy)):
                yield phy[i], res[i]

        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        if shuffle:
            shuffle_buf = min(len(phy), 4096)
            if shuffle_buf > 1:
                ds = ds.shuffle(shuffle_buf, reshuffle_each_iteration=True)
        return ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    return (
        make_dataset(train_phy, train_res, shuffle=True),
        make_dataset(val_phy, val_res),
        train_idx,
        val_idx,
    )
