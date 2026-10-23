import tensorflow as tf

def train_rnn_step(rnn, optimizer, z, actions):

    # 現在状態と次状態に分割    z=(batch_size, timesteps, z_size)
    z1 = z[:, :-1, :]
    z2 = z[:, 1:, :]

    # 行動をone-hotベクトルに変換
    onehot_actions = tf.one_hot(
        actions,
        depth=rnn.action_num
    )

    # tf.GradientTape(): 順伝播と損失計算の演算を記録する --- with文で記録する範囲を指定
    with tf.GradientTape() as tape:

        # MDN-RNNの順伝播
        pi, mu, log_sigma, _ = rnn(
            z1,
            onehot_actions,
            hidden_state=None,
            training=True
        )

        # 混合係数の対数softmax
        log_pi = tf.nn.log_softmax(
            pi, axis=-1
        )

        # 次状態の次元を追加
        z2 = tf.expand_dims(z2, axis=-1)

        # 正規分布の対数確率密度
        log_gauss = -0.5 * (
            tf.math.log(2.0 * tf.constant(3.141592653589793))
            + 2.0 * log_sigma
            + tf.square((z2 - mu) / tf.exp(log_sigma))
        )

        # 混合正規分布の負の対数尤度
        log_prob = tf.reduce_logsumexp(
            log_pi + log_gauss,
            axis=-1
        )

        loss = -tf.reduce_mean(log_prob)

    # 勾配を計算
    grads = tape.gradient(
        loss,
        rnn.trainable_variables
    )

    # 重みを更新 --- 対応するタプルを生成
    optimizer.apply_gradients(
        zip(grads, rnn.trainable_variables)
    )

    return loss