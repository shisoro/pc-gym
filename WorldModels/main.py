import tensorflow as tf
from tensorflow import keras

from config import Config
from MDN_RNN import MDNRNN
from train_RNN import train_rnn_step


def main():
    # 設定
    config = Config()

    # MDN-RNNを生成
    rnn = MDNRNN(config)

    # 最適化アルゴリズム
    optimizer = keras.optimizers.Adam(
        learning_rate=config.learning_rate
    )

    # 仮の学習データ
    z = tf.random.normal((
        config.batch_size,
        config.sequence_length + 1,
        config.z_size
    ))

    actions = tf.random.uniform(
        (config.batch_size, config.sequence_length),
        minval=0,
        maxval=config.action_num,
        dtype=tf.int32
    )

    # 学習ループ
    for epoch in range(100):

        loss = train_rnn_step(
            rnn,
            optimizer,
            z,
            actions
        )

        if epoch % 10 == 0:
            print(f"epoch={epoch}, loss={loss.numpy():.6f}")


if __name__ == "__main__":
    main()