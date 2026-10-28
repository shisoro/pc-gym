
import numpy as np
import tensorflow as tf


def preprocess(obs):
    """観測画像をVAE用に変換"""

    x = tf.convert_to_tensor(obs)

    # RGB画像を64×64へ変換
    x = tf.image.resize(x, (64, 64))

    # 画素値を0〜1に正規化
    x = tf.cast(x, tf.float32) / 255.0

    # バッチ次元を追加
    x = tf.expand_dims(x, axis=0)

    return x


def evaluate_controller(
    env,
    vae,
    rnn,
    controller,
    max_steps=1000
):
    obs, info = env.reset()

    # LSTMの初期状態
    hidden_state = rnn.get_initial_state(
        batch_size=1
    )

    total_reward = 0.0

    for step in range(max_steps):

        # 画像の前処理
        x = preprocess(obs)

        # VAEによる潜在変数の生成
        z = vae.encode(x, training=False)

        # LSTMの隠れ状態
        h, c = hidden_state

        # Controllerによる行動選択
        q = controller(z, h)

        action = int(
            tf.argmax(q[0]).numpy()
        )

        # RNNに入力する行動
        onehot_action = tf.one_hot(
            [action],
            depth=rnn.action_num
        )

        # 時系列次元を追加
        z_seq = tf.expand_dims(z, axis=1)

        action_seq = tf.expand_dims(
            onehot_action, axis=1
        )

        # RNNの隠れ状態を更新
        pi, mu, log_sigma, hidden_state = rnn(
            z_seq,
            action_seq,
            hidden_state,
            training=False
        )

        # 環境を進める
        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += float(reward)

        if terminated or truncated:
            break

    return total_reward


def evaluate_fn(controller):

    obs, info = env.reset()

    hidden_state = rnn.get_initial_state()

    total_reward = 0.0
    done = False

    while not done:

        # 画像の前処理
        x = preprocess(obs)

        # VAEによる潜在変数の生成
        z = vae.encode(x, training=False)

        # LSTMの隠れ状態を取得
        h, c = hidden_state

        # Controllerによる行動決定
        q = controller(z, h)

        action = int(tf.argmax(q[0]).numpy())

        # MDN-RNNの隠れ状態を更新
        # 既存のmdn_rnn.pyのAPIに合わせて実装する
        hidden_state = update_rnn_state(
            rnn, z, action, hidden_state
        )

        # 環境を進める
        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward
        done = terminated or truncated

    return total_reward
