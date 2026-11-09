
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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
    max_steps=1000,
    seed=None,
    record_actions=None
):
    # 環境値を初期化
    obs, info = env.reset(seed=seed)

    # LSTMの初期状態
    hidden_state = rnn.get_initial_state(
        batch_size=1
    )

    total_reward = 0.0

    # Controllerの性能記録
    action_history = []

    for step in range(max_steps):

        # 画像の前処理
        x = preprocess(obs)

        # VAEによる潜在変数の生成（潜在変数 z = μ に設定）
        ## z = vae.encode(x, training=False) 
        z_mean, z_log_var = vae.encoder(x, training=False)
        z = z_mean

        # Controllerで連続行動を決定
        action = controller(z, hidden_state) 

        # TensorFlowのTensorからNumPy配列に変換
        action = action.numpy()[0].astype(np.float32)

        # ――― Controllerの性能を確認 ―――
        action_history.append(action)
        
        if step % 50 == 0:
            print(
                f"Step {step}: "
                f"steering={action[0]:.3f}, "
                f"gas={action[1]:.3f}, "
                f"brake={action[2]:.3f}"
            )
        # ―――――――――――――――――――――――――――

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
    
    if record_actions == True:
        steps = np.arange(len(action_history))

        plt.figure(figsize=(12, 5))

        plt.plot(steps, action_history[:, 0], label="Steering")
        plt.plot(steps, action_history[:, 1], label="Gas")
        plt.plot(steps, action_history[:, 2], label="Brake")

        plt.xlabel("Step")
        plt.ylabel("Action value")
        plt.title("Controller action history")
        plt.legend()
        plt.grid(True)

        plt.savefig("action_history.png")
        plt.close()

    print("\n--- Action statistics ---")
    print("Steering mean:", np.mean(action_history[:, 0]))
    print("Gas mean:", np.mean(action_history[:, 1]))
    print("Brake mean:", np.mean(action_history[:, 2]))
    print("Gas max:", np.max(action_history[:, 1]))
    print("Brake max:", np.max(action_history[:, 2]))

    # アクセルとブレーキの同時使用率を調べる
    simultaneous = np.logical_and(
    action_history[:, 1] > 0.8,
    action_history[:, 2] > 0.8
    )

    print("Simultaneous gas/brake rate:", np.mean(simultaneous))
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
