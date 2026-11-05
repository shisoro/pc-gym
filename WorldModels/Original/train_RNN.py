import tensorflow as tf

def train_rnn_step(rnn, optimizer, z, actions):

    # 現在状態と次状態に分割    z=(batch_size, timesteps, z_size)
    z1 = z[:, :-1, :]
    z2 = z[:, 1:, :]

    # tf.GradientTape(): 順伝播と損失計算の演算を記録する --- with文で記録する範囲を指定
    with tf.GradientTape() as tape:

        # MDN-RNNの順伝播
        pi, mu, log_sigma, _ = rnn(
            z1,
            actions,
            hidden_state=None,
            training=True
        )

        tf.print("pi range:",
                tf.reduce_min(pi), tf.reduce_max(pi))

        tf.print("mu range:",
                tf.reduce_min(mu), tf.reduce_max(mu))

        tf.print("log_sigma range:",
                tf.reduce_min(log_sigma), tf.reduce_max(log_sigma))

        log_pi = tf.nn.log_softmax(pi, axis=-1)

        z2 = tf.expand_dims(z2, axis=-1)

        # 標準偏差
        sigma = tf.exp(log_sigma)

        # ガウス分布の対数確率密度
        normalized_error = (z2 - mu) / sigma

        log_prob = (
            -0.5 * tf.math.log(2.0 * tf.constant(3.141592653589793))
            - log_sigma
            - 0.5 * tf.square(normalized_error) #tf.square(normalized_error)の係数を+1から-0.5に変更（計算式にミスがあったらしい）
        )

        log_mix = tf.reduce_logsumexp(
            log_pi + log_prob,
            axis=-1
        )

        loss = -tf.reduce_mean(log_mix)

    # 勾配を計算
    grads = tape.gradient(
        loss,
        rnn.trainable_variables
    )

    # 損失と勾配が有限値か確認
    tf.debugging.assert_all_finite(loss, "Loss is NaN or Inf")

    for i, grad in enumerate(grads):
        if grad is not None:
            tf.debugging.assert_all_finite(
                grad,
                f"Gradient {i} is NaN or Inf"
            )

    # 勾配全体の大きさを確認
    global_norm = tf.linalg.global_norm(grads)

    tf.print("NLL:", loss)
    tf.print("Gradient global norm:", global_norm)

    # 重みを更新 --- 対応するタプルを生成
    optimizer.apply_gradients(
        zip(grads, rnn.trainable_variables)
    )

    return loss