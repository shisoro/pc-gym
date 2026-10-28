import tensorflow as tf

def train_vae_step(vae, optimizer, x):

    with tf.GradientTape() as tape:

        # VAEのEncoder
        z_mean, z_log_var, z = vae.encode(
            x, training=True
        )

        # VAEのDecoder
        pred_x = vae.decode(
            z, training=True
        )

        # Reconstruction loss --- 出力結果(batch_size,)        
        rc_loss = tf.reduce_sum(
            tf.square(x - pred_x),
            axis=[1, 2, 3]
        )

        # KL divergence 
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var
            - tf.square(z_mean)
            - tf.exp(z_log_var),
            axis=1
        )

        kl_tolerance = 0.5
        # KL損失を強く最小化しすぎるとEncoderが入力画像の情報を潜在変数に十分反映しなくなる
        kl_loss = tf.maximum(
            kl_loss,
            kl_tolerance * vae.z_size
        )

        kl_loss = tf.reduce_mean(kl_loss)

        # Total loss
        loss = rc_loss + kl_loss

    # Gradient
    grads = tape.gradient(
        loss,
        vae.trainable_variables
    )

    # Update weights
    optimizer.apply_gradients(
        zip(grads, vae.trainable_variables)
    )

    return loss, rc_loss, kl_loss