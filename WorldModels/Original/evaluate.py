
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
    # 環境値を初期化
    obs, info = env.reset()

    # LSTMの初期状態
    hidden_state = rnn.get_initial_state(
        batch_size=1
    )

    total_reward = 0.0

    # ――― Controllerの性能を確認 ―――
    action_history = []

    # ループ内のControllerによる行動決定部分を変更
    action = controller(z, hidden_state)
    action = action.numpy()[0].astype(np.float32)

    action_history.append(action)

    if step % 50 == 0:
        print(
            f"Step {step}: "
            f"steering={action[0]:.3f}, "
            f"gas={action[1]:.3f}, "
            f"brake={action[2]:.3f}"
        )
    # ―――――――――――――――――――――――――――

    for step in range(max_steps):

        # 画像の前処理
        x = preprocess(obs)

        # VAEによる潜在変数の生成
        z = vae.encode(x, training=False)

        # Controllerで連続行動を決定
        action = controller(z, hidden_state) 

        # TensorFlowのTensorからNumPy配列に変換
        action = action.numpy()[0].astype(np.float32)

        # MDN-RNNの隠れ状態を更新
        hidden_state = rnn.rnn(
            z,
            action.reshape(1, 3),
            hidden_state,
            rnn_only=True
        )

        # 環境を進める
        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += float(reward)

        if terminated or truncated:
            break

        # ――― Controllerの性能を表示 ―――
        action_history = np.array(action_history)

        print("\n--- Action statistics ---")
        print("Steering mean:", np.mean(action_history[:, 0]))
        print("Gas mean:", np.mean(action_history[:, 1]))
        print("Brake mean:", np.mean(action_history[:, 2]))
        print("Gas max:", np.max(action_history[:, 1]))
        print("Brake max:", np.max(action_history[:, 2]))
        # ―――――――――――――――――――――――――――

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
