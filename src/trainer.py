import tensorflow as tf
import time
from tqdm import tqdm
import numpy as np
import os
from types import SimpleNamespace
from .losses import encoder_loss, discriminator_loss
from .masking import mix_features
namespace = tf.keras

class EBNL_Trainer:
    def __init__(self, encoder, decoder, discriminator, res_discriminator, config, topology):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.res_discriminator = res_discriminator
        if topology is None:
            topology = SimpleNamespace(
                idx_res=np.array([]),
                idx_phy=np.array([]),
                res_to_dead_local=[],
                res_to_lone_local=[],
            )
        self.topo = topology 

        # Config initialization
        self.latent_dim = config["latent_dim"]
        self.lr = config["lr"]
        self.lambda_d = config["lambda_d"]
        self.lambda_e = config["lambda_e"]
        self.recon_weight = config["recon_weight"]
        self.res_adv_weight = config.get("res_adv_weight", 1.0)
        self.jitter_alpha_max = config.get("jitter_alpha_max", 0.45)
        self.lambda_joint = config.get("lambda_joint", 1.0)
        self.patience = config["patience"]
        self.batch_size = config["batch_size"]

        self.hnn_optimizer = namespace.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)
        self.fno_optimizer = namespace.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)
        self.recon_metric = namespace.metrics.Mean(name="recon_mean")
        self.disc_metric = namespace.metrics.Mean(name="disc_mean")
        self.enc_metric = namespace.metrics.Mean(name="enc_mean")

        res_dim = len(self.topo.idx_res)
        iso_mask = np.zeros((res_dim,), dtype=np.float32)
        if len(self.topo.res_to_lone_local) > 0:
            iso_mask[self.topo.res_to_lone_local] = 1.0
        self.res_isolate_mask = tf.constant(iso_mask.reshape(1, 1, -1), dtype=tf.float32)

    def _append_time(self, res_windows):
        batch = tf.shape(res_windows)[0]
        steps = tf.shape(res_windows)[1]
        t = tf.cast(tf.range(steps), tf.float32)[tf.newaxis, :, tf.newaxis]
        t = tf.broadcast_to(t, [batch, steps, 1])
        return tf.concat([res_windows, t], axis=-1)

    @tf.function
    def _train_step(self, phy_packed, res_windows):
        anchor_phy = phy_packed[:, 0, :, :]        
        aug_views_phy = phy_packed[:, 1:5, :, :] 
        
        with tf.GradientTape(persistent=True) as tape:
            # 2. Forward Pass
            res_with_time = self._append_time(res_windows)
            z_sys, z_res, _ = self.encoder([anchor_phy, res_with_time], training=True)
            recons_phy, recons_res = self.decoder([z_sys, z_res], training=True)
            
            # 3. HNN Reconstruction Loss
            if self.topo.idx_phy.shape[0] > 0:
                recon_phy_loss = tf.reduce_mean(tf.square(anchor_phy - recons_phy))
            else:
                recon_phy_loss= tf.constant(0.0, dtype=tf.float32)

            # 4. FNO reconstruction loss
            recon_res_loss = tf.reduce_mean(tf.square(res_windows - recons_res))
            total_recon_loss = recon_phy_loss + recon_res_loss

            # 5. HNN adversarial flow:
            # mix the anchor physics latent with augmented-view latents, then train
            # the discriminator to separate matched vs shuffled latent mixtures.
            z_pos_all, alpha_all = [], []
            for i in range(4):
                view_phy = aug_views_phy[:, i, :, :]
                z_s_aug, _, _ = self.encoder([view_phy, res_with_time], training=True)
                z_mixed, alpha = mix_features(z_sys, z_s_aug)
                z_pair = tf.concat([z_sys, z_mixed], axis=1) 
                
                z_pos_all.append(z_pair)
                alpha_all.append(alpha)

            z_mixed_pos = tf.concat(z_pos_all, axis=0)
            alpha_pos = tf.concat(alpha_all, axis=0)
            
            z_neg_mixed, beta_neg = mix_features(z_sys, tf.random.shuffle(z_sys))
            z_neg_pair = tf.concat([z_sys, z_neg_mixed], axis=1) 

            d_out_pos = self.discriminator(z_mixed_pos, training=True)
            d_out_neg = self.discriminator(z_neg_pair, training=True)

            loss_disc = discriminator_loss(d_out_pos, d_out_neg, alpha_pos, beta_neg)
            loss_enc = encoder_loss(d_out_pos, d_out_neg, beta_neg)

            # 6. FNO adversarial flow:
            # jitter the appended time grid on isolated residual channels and train
            # the residual discriminator to recover the applied jitter strength.
            res_adv_loss = tf.constant(0.0, dtype=tf.float32)
            res_disc_loss = tf.constant(0.0, dtype=tf.float32)
            if tf.reduce_sum(self.res_isolate_mask) > 0:
                res_iso = res_windows * self.res_isolate_mask
                batch = tf.shape(res_windows)[0]
                steps = tf.shape(res_windows)[1]
                t = tf.cast(tf.range(steps), tf.float32)[tf.newaxis, :, tf.newaxis]
                t = tf.broadcast_to(t, [batch, steps, 1])
                alpha = tf.random.uniform((batch, 1, 1), 0.0, self.jitter_alpha_max)
                eps = tf.random.uniform((batch, steps, 1), -alpha, alpha)
                t_jit = t + eps
                res_iso_time = tf.concat([res_iso, t_jit], axis=-1)

                anchor_stop = tf.stop_gradient(anchor_phy)
                _, z_res_jit, _ = self.encoder([anchor_stop, res_iso_time], training=True)
                alpha_pred = self.res_discriminator(z_res_jit, training=True)
                alpha_target = tf.reshape(alpha, (-1, 1))
                res_disc_loss = tf.reduce_mean(tf.square(alpha_pred - alpha_target))
                res_adv_loss = tf.reduce_mean(tf.square(alpha_pred))

            hnn_total = (self.recon_weight * recon_phy_loss +
                         self.lambda_d * loss_disc +
                         self.lambda_e * loss_enc)

            fno_total = (self.recon_weight * recon_res_loss +
                         self.res_adv_weight * res_adv_loss +
                         res_disc_loss +
                         self.lambda_joint * loss_enc)

        # Apply two optimizer views over the shared encoder/decoder: one driven by
        # the HNO adversarial objective, one by the residual objective.
        hnn_vars = (
            self.encoder.trainable_variables +
            self.decoder.trainable_variables +
            self.discriminator.trainable_variables
        )
        fno_vars = (
            self.encoder.trainable_variables +
            self.decoder.trainable_variables +
            self.res_discriminator.trainable_variables
        )

        hnn_grads = tape.gradient(hnn_total, hnn_vars)
        fno_grads = tape.gradient(fno_total, fno_vars)

        hnn_pairs = [(g, v) for g, v in zip(hnn_grads, hnn_vars) if g is not None]
        fno_pairs = [(g, v) for g, v in zip(fno_grads, fno_vars) if g is not None]

        if hnn_pairs:
            self.hnn_optimizer.apply_gradients(hnn_pairs)
        if fno_pairs:
            self.fno_optimizer.apply_gradients(fno_pairs)
        
        return total_recon_loss, loss_disc, loss_enc

    def fit(self, train_ds, val_ds=None, epochs=200, steps_per_epoch=None):
        best_val_loss = float("inf")
        wait = 0 
        epoch_times = []
        start_time = time.perf_counter()
        converge_time = None
        
        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            self.recon_metric.reset_state()
            self.disc_metric.reset_state()
            self.enc_metric.reset_state()
            
            bar = tqdm(
                train_ds,
                total=steps_per_epoch,
                desc=f"Epoch {epoch+1}",
                unit="batch",
            )
            for phy_batch, res_batch in bar:
                r_loss, d_loss, e_loss = self._train_step(phy_batch, res_batch)
                self.recon_metric.update_state(r_loss)
                self.disc_metric.update_state(d_loss)
                self.enc_metric.update_state(e_loss)

                
                bar.set_postfix({"Recon": f"{r_loss:.4f}", "Disc": f"{d_loss:.4f}"})

            if val_ds is not None:
                val_losses = []
                for v_phy, v_res in val_ds:
                    v_phy_anchor = v_phy[:, 0, :, :] 
                    v_res_time = self._append_time(v_res)
                    z_s, z_r, _ = self.encoder([v_phy_anchor, v_res_time], training=False)
                    h_phy, h_res = self.decoder([z_s, z_r], training=False)
                    if self.topo.idx_phy.shape[0] > 0:
                        mse_phy = tf.reduce_mean(tf.square(v_phy_anchor - h_phy))
                    else:
                        mse_phy = tf.constant(0.0, dtype=tf.float32)

                    if self.topo.idx_res.shape[0] > 0:
                        mse_res = tf.reduce_mean(tf.square(v_res - h_res))
                    else:
                        mse_res = tf.constant(0.0, dtype=tf.float32)
                    val_losses.append((mse_phy + mse_res).numpy())
                
                current_val_loss = float(np.mean(val_losses))
                print(f"Val MSE: {current_val_loss:.4f} | Best: {best_val_loss:.4f}")
                
                if current_val_loss < best_val_loss:
                    best_val_loss, wait = current_val_loss, 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        converge_time = time.perf_counter() - start_time
                        break
            epoch_times.append(time.perf_counter() - epoch_start)

        total_time = time.perf_counter() - start_time
        if converge_time is None:
            converge_time = total_time

        return {
            "epoch_times": epoch_times,
            "total_time": total_time,
            "converge_time": converge_time,
            "epochs_ran": len(epoch_times),
        }
                    
    def reconstruct(self, test_final, batch_size=128):
        print(" window reconstruction")
        phy_views = np.asarray(test_final['phy'], dtype=np.float32)
        res_data = np.asarray(test_final['res'], dtype=np.float32)
        phy_anchor = phy_views[:, 0, :, :] if len(phy_views.shape) == 4 else phy_views

        phy_hat_chunks, res_hat_chunks = [], []
        num_batches = (len(phy_anchor) + batch_size - 1) // batch_size
        recon_bar = tqdm(
            range(0, len(phy_anchor), batch_size),
            total=num_batches,
            desc="Reconstruct",
            unit="batch",
        )
        for i in recon_bar:
            p_batch = tf.convert_to_tensor(phy_anchor[i:i+batch_size], dtype=tf.float32)
            r_batch = tf.convert_to_tensor(res_data[i:i+batch_size], dtype=tf.float32)
            r_batch_time = self._append_time(r_batch)
            zs, zr, _ = self.encoder([p_batch, r_batch_time], training=False)
            ph, rh = self.decoder([zs, zr], training=False)
            phy_hat_chunks.append(ph.numpy())
            res_hat_chunks.append(rh.numpy())
            recon_bar.set_postfix({"done": f"{min(i + batch_size, len(phy_anchor))}/{len(phy_anchor)}"})

        return {
            "phy_orig": phy_anchor,
            "phy_hat": np.concatenate(phy_hat_chunks, axis=0),
            "res_orig": res_data,
            "res_hat": np.concatenate(res_hat_chunks, axis=0),
        }

    def reconstruct_stitched(
        self,
        data_final,
        *,
        batch_size=128,
        window_size,
        stride,
        total_len,
        window_indices=None,
        save_dir=None,
        split=None,
        labels=None,
    ):
        print(" window reconstruction")
        phy_views = np.asarray(data_final["phy"], dtype=np.float32)
        res_data = np.asarray(data_final["res"], dtype=np.float32)
        phy_anchor = phy_views[:, 0, :, :] if len(phy_views.shape) == 4 else phy_views

        phy_dim = phy_anchor.shape[-1] if phy_anchor.ndim == 3 else 0
        res_dim = res_data.shape[-1] if res_data.ndim == 3 else 0
        num_windows = len(phy_anchor)

        stitched_phy = np.zeros((total_len, phy_dim), dtype=np.float32) if phy_dim > 0 else np.zeros((total_len, 0), dtype=np.float32)
        stitched_res = np.zeros((total_len, res_dim), dtype=np.float32) if res_dim > 0 else np.zeros((total_len, 0), dtype=np.float32)
        counts = np.zeros(total_len, dtype=np.float32)

        phy_raw = None
        res_raw = None
        phy_tmp = None
        res_tmp = None
        if save_dir and split:
            os.makedirs(save_dir, exist_ok=True)
            if phy_dim > 0:
                phy_tmp = os.path.join(save_dir, f".{split}_phy_raw.tmp.npy")
                phy_raw = np.lib.format.open_memmap(phy_tmp, mode="w+", dtype=np.float32, shape=phy_anchor.shape)
            if res_dim > 0:
                res_tmp = os.path.join(save_dir, f".{split}_res_raw.tmp.npy")
                res_raw = np.lib.format.open_memmap(res_tmp, mode="w+", dtype=np.float32, shape=res_data.shape)

        prev_w_idx = None
        num_batches = (num_windows + batch_size - 1) // batch_size
        recon_bar = tqdm(
            range(0, num_windows, batch_size),
            total=num_batches,
            desc="Reconstruct",
            unit="batch",
        )
        for start_idx in recon_bar:
            end_idx = min(start_idx + batch_size, num_windows)
            p_batch = tf.convert_to_tensor(phy_anchor[start_idx:end_idx], dtype=tf.float32)
            r_batch = tf.convert_to_tensor(res_data[start_idx:end_idx], dtype=tf.float32)
            r_batch_time = self._append_time(r_batch)
            zs, zr, _ = self.encoder([p_batch, r_batch_time], training=False)
            ph, rh = self.decoder([zs, zr], training=False)

            ph_np = ph.numpy() if phy_dim > 0 else None
            rh_np = rh.numpy() if res_dim > 0 else None
            phy_err = np.square(phy_anchor[start_idx:end_idx] - ph_np, dtype=np.float32) if phy_dim > 0 else None
            res_err = np.square(res_data[start_idx:end_idx] - rh_np, dtype=np.float32) if res_dim > 0 else None

            if phy_raw is not None and phy_err is not None:
                phy_raw[start_idx:end_idx] = phy_err
            if res_raw is not None and res_err is not None:
                res_raw[start_idx:end_idx] = res_err

            for local_i in range(end_idx - start_idx):
                global_i = start_idx + local_i
                w_idx = int(window_indices[global_i]) if window_indices is not None else global_i
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
                if phy_err is not None:
                    stitched_phy[slice_start:end] += phy_err[local_i, offset: offset + slice_len]
                if res_err is not None:
                    stitched_res[slice_start:end] += res_err[local_i, offset: offset + slice_len]
                counts[slice_start:end] += 1
                prev_w_idx = w_idx

            del p_batch, r_batch, r_batch_time, zs, zr, ph, rh, ph_np, rh_np, phy_err, res_err
            recon_bar.set_postfix({"done": f"{end_idx}/{num_windows}"})

        denom = np.maximum(counts[:, None], 1.0)
        out = {
            "stitched_phy": stitched_phy / denom if phy_dim > 0 else stitched_phy,
            "stitched_res": stitched_res / denom if res_dim > 0 else stitched_res,
        }

        if save_dir and split:
            np.savez_compressed(
                os.path.join(save_dir, f"{split}_meta.npz"),
                window_indices=np.asarray(window_indices) if window_indices is not None else None,
                total_len=int(total_len),
                stride=int(stride),
                window_size=int(window_size),
                labels=np.asarray(labels) if labels is not None else None,
            )
            if phy_raw is not None:
                phy_raw.flush()
                np.savez_compressed(os.path.join(save_dir, f"{split}_phy_raw.npz"), err=phy_raw)
                del phy_raw
                os.remove(phy_tmp)
            if res_raw is not None:
                res_raw.flush()
                np.savez_compressed(os.path.join(save_dir, f"{split}_res_raw.npz"), err=res_raw)
                del res_raw
                os.remove(res_tmp)

        return out
