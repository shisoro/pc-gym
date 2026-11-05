
import glob
import os

import numpy as np
import tensorflow as tf

from config import Config
from MDN_RNN import _MDNRNN
from train_RNN import train_rnn_step


def load_rnn_data(sequence_length):

    z_windows = []
    action_windows = []

    # VAEによって変換したデータを取得
    files = sorted(
        glob.glob("data/rnn/episode_*.npz")
    )

    if not files:
        raise FileNotFoundError(
            "data/rnn/ に学習データがありません"
        )

    for file in files:

        with np.load(file) as data:
            z = data["z"].astype(np.float32)
            actions = data["actions"].astype(np.float32)

        # 状態数 = 行動数 + 1 であることを確認
        if len(z) != len(actions) + 1:
            raise ValueError(
                f"状態と行動の数が一致しません: {file}"
            )

        # エピソード内で時系列を分割
        # 他のエピソードとの境界はまたがない
        for start in range(
            0,
            len(actions) - sequence_length + 1,
            sequence_length
        ):

            end = start + sequence_length

            # 11個の潜在変数
            z_window = z[start:end + 1]

            # 10個の行動
            action_window = actions[start:end]

            z_windows.append(z_window)
            action_windows.append(action_window)

    if not z_windows:
        raise ValueError(
            "時系列データを作成できませんでした"
        )    

    return (
        np.stack(z_windows),
        np.stack(action_windows)
    )


def main():

    config = Config()

    # --- 学習データの準備
    z, actions = load_rnn_data(
        config.sequence_length
    )

    print("z shape:", z.shape)
    print("actions shape:", actions.shape)

    dataset = tf.data.Dataset.from_tensor_slices(
        (z, actions)
    )

    dataset = (
        dataset
        .shuffle(buffer_size=len(z)) # 時系列サンプルの順番をランダムに並び替える
        .batch(config.batch_size) 
        .prefetch(tf.data.AUTOTUNE)
    )

    # --- MDN-RNNの生成
    rnn = _MDNRNN(config)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate
    )

    # モデル全体を構築
    # save_weights()で未構築エラーが出ないようにする
    dummy_z = tf.zeros(
        (1, config.sequence_length, config.z_size)
    )

    dummy_actions = tf.zeros(
        (1, config.sequence_length, config.action_num)
    )

    rnn(
        dummy_z,
        dummy_actions,
        hidden_state=None,
        training=False
    )

    print("RNN built:", rnn.built)

    # --- 学習
    epochs = 50

    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):

        total_loss = 0.0
        total_samples = 0

        for batch_z, batch_actions in dataset:

            loss = train_rnn_step(
                rnn,
                optimizer,
                batch_z,
                batch_actions
            )

            batch_size = int(tf.shape(batch_z)[0])

            total_loss += float(loss.numpy()) * batch_size
            total_samples += batch_size

        average_loss = total_loss / total_samples

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"NLL: {average_loss:.6f}"
        )

        # 10エポックごとに保存
        if (epoch + 1) % 10 == 0:

            rnn.save_weights(
                "models/mdn_rnn.weights.h5"
            )

    # 最終的な重みを保存
    rnn.save_weights(
        "models/mdn_rnn.weights.h5"
    )

    print("MDN-RNNの学習が完了しました")


if __name__ == "__main__":
    main()
