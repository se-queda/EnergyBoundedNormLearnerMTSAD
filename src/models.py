import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


namespace = tf.keras

REG = 1e-4

    

def hnn_projector(x, projection_dim=256):
    for i in range(5):
        dilation = 2**i
        shortcut = x
        if x.shape[-1] != projection_dim:
            shortcut = layers.Conv1D(projection_dim, 1, padding='same')(x)
        x = layers.Conv1D(
            filters=projection_dim,
            kernel_size=3,
            dilation_rate=dilation,
            padding='causal', 
            activation='tanh', 
            kernel_regularizer=regularizers.l2(REG)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('tanh')(x)
        
    return x # Shape: (Batch, 64, 256)

class BiDirectionalSymplecticLayer(layers.Layer):
    def __init__(self, feature_dim, steps, dt, **kwargs):
        super(BiDirectionalSymplecticLayer, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.steps = steps
        self.dt = dt
        # iinternal Hamiltonian MLP 
        self.h_dense1 = layers.Dense(feature_dim * 2, activation='tanh') 
        self.h_dense2 = layers.Dense(feature_dim * 2, activation='tanh')
        self.h_out = layers.Dense(1, use_bias=False, name="h_out") 

    def get_gradients(self, q, p):
        state_p = tf.concat([q, p], axis=-1)
        with tf.GradientTape() as tape:
            tape.watch(state_p)
            x = self.h_dense1(state_p)
            x = self.h_dense2(x)
            H = self.h_out(x)
        dH = tape.gradient(H, state_p)
        # Returns dq (dH/dp) and dp (-dH/dq)
        return dH[:, self.feature_dim:], -dH[:, :self.feature_dim]

    def leapfrog_step(self, q, p, dt):
        # Standard Symplectic Leapfrog Step
        dq, dp = self.get_gradients(q, p)
        p = p + 0.5 * dt * dp
        q = q + dt * dq
        _, dp_final = self.get_gradients(q, p)
        p = p + 0.5 * dt * dp_final
        return q, p

    def call(self, x):
        # 1. Anchor at the Midpoint (Center of the 64-step window)
        mid_idx = x.shape[1] // 2 
        
        # Extract coordinates at the center
        q_mid = x[:, mid_idx, :] 
        p_mid = x[:, mid_idx, :] - x[:, mid_idx - 1, :] 
        
        # 2. Forward Integration (+dt) from center to end
        q_f, p_f = q_mid, p_mid
        for _ in range(self.steps // 2):
            q_f, p_f = self.leapfrog_step(q_f, p_f, self.dt)
            
        # 3. Backward Integration (-dt) from center to start
        q_b, p_b = q_mid, p_mid
        for _ in range(self.steps // 2):
            q_b, p_b = self.leapfrog_step(q_b, p_b, -self.dt)
            
        return tf.concat([q_b, p_b, q_mid, p_mid, q_f, p_f], axis=-1)


def fno_lifting(x, latent_dim=256):
    x = layers.Dense(latent_dim, 
                     kernel_regularizer=regularizers.l2(REG),
                     name="FNO_Lifting")(x)
    x = layers.Activation('gelu')(x) 
    return x
class SpectralConv1D(layers.Layer):
    def __init__(self, out_channels, modes, **kwargs):
        super(SpectralConv1D, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.modes = modes

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.weights_real = self.add_weight(
            shape=(self.modes, in_channels, self.out_channels),
            initializer='glorot_normal',
            trainable=True,
            name='spectral_weights_real'
        )
        self.weights_imag = self.add_weight(
            shape=(self.modes, in_channels, self.out_channels),
            initializer='zeros',
            trainable=True,
            name='spectral_weights_imag'
        )

    def call(self, x):
        n = tf.shape(x)[1]
        
        # 1. Fourier Transform
        x = tf.transpose(x, perm=[0, 2, 1]) 
        x_ft = tf.signal.rfft(x) 
        
        # 2. Spectral Filter
        x_ft_low = x_ft[:, :, :self.modes]
        weights = tf.complex(self.weights_real, self.weights_imag)
        out_ft_low = tf.einsum('bim,mio->bom', x_ft_low, weights)
        
        # 3. Inverse Transform & Resolution Recovery
        padding = tf.zeros([tf.shape(out_ft_low)[0], self.out_channels, (n // 2 + 1) - self.modes], dtype=tf.complex64)
        out_ft = tf.concat([out_ft_low, padding], axis=-1)
        
        x = tf.signal.irfft(out_ft, fft_length=[n])
        x = tf.transpose(x, perm=[0, 2, 1])
        x.set_shape([None, None, self.out_channels])
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.out_channels)

def fno_block(x, filters, modes, dropout=0.1):
    shortcut = namespace.layers.Conv1D(filters, 1, padding='same', 
                                      kernel_regularizer=namespace.regularizers.l2(REG))(x)
    
    x_f = SpectralConv1D(filters, modes)(x)
    x_f = namespace.layers.BatchNormalization()(x_f)
    
    x = namespace.layers.Add()([x_f, shortcut])
    x = namespace.layers.Activation('gelu')(x)
    x = namespace.layers.SpatialDropout1D(dropout)(x)
    return x


def build_dual_encoder(input_shape_sys, input_shape_res, config):
    L = config.get("latent_dim", 512) 
    h_dim = config.get("hnn_feature_dim", 256) 
    drop_rate = config.get("dropout", 0.1)
    feat_sys = input_shape_sys[1]
    inputs_sys = layers.Input(shape=input_shape_sys, name="input_sys")
    inputs_res = layers.Input(shape=input_shape_res, name="input_res")
    f_modes = config.get("fno_modes", 32)
    
    if feat_sys>0:
    # Branch A: Hamiltonian
        proj_hnn = hnn_projector(inputs_sys, projection_dim=h_dim)
        flow = BiDirectionalSymplecticLayer(
            feature_dim=h_dim, 
            steps=config.get("hnn_steps", 4), 
            dt=config.get("hnn_dt", 0.1)
        )(proj_hnn)
        x_sys = layers.Dense(L // 2, activation='tanh', 
                                kernel_regularizer=regularizers.l2(REG))(flow)
        x_sys = layers.Dropout(drop_rate)(x_sys)
        z_sys = layers.Dense(L // 2, name="z_sys")(x_sys)
    else:
        z_sys = layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], L // 2)), name="z_sys")(inputs_sys)
    
    # Branch B: FNO Encoder
    # 1. Lifting Layer
    x_fno = fno_lifting(inputs_res, config.get("hnn_feature_dim", 256))
    for _ in range(config.get("fno_blocks", 1)):
        x_fno = fno_block(x_fno, h_dim, modes=f_modes)
    
    # 3. Final Latent Projection
    x_fno = layers.GlobalAveragePooling1D()(x_fno)
    z_res = layers.Dense(L // 2, name="z_res")(x_fno)
    
    z_combined = layers.Concatenate(name="z_combined")([z_sys, z_res])
    return models.Model([inputs_sys, inputs_res], [z_sys, z_res, z_combined])


def build_dual_decoder(feat_sys, feat_res, output_steps, config):
    L = config.get("latent_dim", 512)
    h_dim = config.get("hnn_feature_dim", 256)
    f_modes = config.get("fno_modes", 8)
    drop_rate = config.get("dropout", 0.1)
    
    # Latent inputs
    z_sys_in = layers.Input(shape=(L // 2,), name="z_sys_input")
    z_res_in = layers.Input(shape=(L // 2,), name="z_res_input")
    
    # DECODER A:Hamiltonian Reconstruction
    x_s = layers.Dense(output_steps * 64, activation='tanh', 
                       kernel_regularizer=regularizers.l2(REG))(z_sys_in)
    x_s = layers.Reshape((output_steps, 64))(x_s)
    
    if feat_sys > 0:
        x_s = layers.Conv1D(feat_sys, 1, padding='same', 
                        kernel_regularizer=regularizers.l2(REG))(x_s)
        out_sys = layers.Activation('linear', name='out_phy')(x_s)
    else:
        # Safety for zero-hnn feature cases
        out_sys = layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], output_steps, 0)), 
                                name='out_phy')(x_s) 
    
    # DECODER B: FNO Path 
    x_r = layers.Dense(output_steps * h_dim, activation='gelu',
                       kernel_regularizer=regularizers.l2(REG))(z_res_in)
    x_r = layers.Reshape((output_steps, h_dim))(x_r)
    for i in range(config.get("fno_decoder_blocks", 1)):
        x_r = fno_block(x_r, h_dim, modes=f_modes, dropout=drop_rate)

    x_r = layers.Conv1D(feat_res, 1, padding='same', 
                        kernel_regularizer=regularizers.l2(REG))(x_r)
    out_res = layers.Activation('linear', name='out_res')(x_r)
    
    return models.Model([z_sys_in, z_res_in], [out_sys, out_res])


def build_discriminator(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation=None)(x)
    return models.Model(inputs, outputs)


def build_res_discriminator(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation=None)(x)
    return models.Model(inputs, outputs)
