import tensorflow as tf
import time
from tqdm import tqdm
import numpy as np
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
        # 1. Unpack Physics Views
        anchor_phy = phy_packed[:, 0, :, :]        
        aug_views_phy = phy_packed[:, 1:5, :, :] 
        
        with tf.GradientTape(persistent=True) as tape:
            # 2. Forward Pass
            res_with_time = self._append_time(res_windows)
            z_sys, z_res, _ = self.encoder([anchor_phy, res_with_time], training=True)
            recons_phy, recons_res = self.decoder([z_sys, z_res], training=True)
            
            # 3. Physics Loss (Hamiltonian Branch)
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

    def fit(self, train_ds, val_ds=None, epochs=200):
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
            
            bar = tqdm(train_ds, desc=f"Epoch {epoch+1}", unit="batch")
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
        print(" Generating Point-wise Reconstructions...")
        phy_views = tf.cast(test_final['phy'], tf.float32) 
        res_data  = tf.cast(test_final['res'], tf.float32)
        phy_anchor = phy_views[:, 0, :, :] if len(phy_views.shape) == 4 else phy_views
        z_sys_list, z_res_list = [], []
        for i in range(0, len(phy_anchor), batch_size):
            p_batch = phy_anchor[i:i+batch_size]
            r_batch = res_data[i:i+batch_size]
            r_batch_time = self._append_time(r_batch)
            zs, zr, _ = self.encoder([p_batch, r_batch_time], training=False)
            z_sys_list.append(zs)
            z_res_list.append(zr)
        
        z_sys = tf.concat(z_sys_list, axis=0)
        z_res = tf.concat(z_res_list, axis=0)
        phy_hat_list, res_hat_list = [], []
        for i in range(0, len(z_sys), batch_size):
            ph, rh = self.decoder([z_sys[i:i+batch_size], z_res[i:i+batch_size]], training=False)
            phy_hat_list.append(ph)
            res_hat_list.append(rh)
            
        return {
            "phy_orig": phy_anchor.numpy(),
            "phy_hat": tf.concat(phy_hat_list, axis=0).numpy(),
            "res_orig": res_data.numpy(),
            "res_hat": tf.concat(res_hat_list, axis=0).numpy()
        }
