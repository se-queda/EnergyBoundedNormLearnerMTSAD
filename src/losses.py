import tensorflow as tf


def discriminator_loss(d_pos, d_neg, alpha, beta):
    target_pos = tf.concat([tf.ones_like(alpha), alpha], axis=1) 
    target_neg = tf.concat([tf.zeros_like(beta), beta], axis=1)  

    loss_pos = tf.reduce_mean(tf.square(d_pos - target_pos))
    loss_neg = tf.reduce_mean(tf.square(d_neg - target_neg))

    return loss_pos + loss_neg


def encoder_loss(d_pos, d_neg, beta):

    fake_alpha = tf.ones_like(d_pos[:, 1:])  # shape: (batch, 1)
    target_pos = tf.concat([tf.ones_like(fake_alpha), fake_alpha], axis=1)  
    target_neg = tf.concat([tf.zeros_like(beta), beta], axis=1)             

    loss_pos = tf.reduce_mean(tf.square(d_pos - target_pos))
    loss_neg = tf.reduce_mean(tf.square(d_neg - target_neg))

    return loss_pos + loss_neg
