
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as kl


class _Controller(keras.Model):

    def __init__(self, config):
        super().__init__()

        self.z_size = config.z_size
        self.hidden_size = config.rnn_units
        self.action_num = config.action_num

        # 潜在変数とLSTM隠れ状態から3つの行動値を出力
        self.fc = kl.Dense(self.action_num)

    def call(self, z, hidden_state):

        # LSTMの隠れ状態 [h, c] から h を取り出す
        h = hidden_state[0]

        # 潜在変数と隠れ状態を結合
        x = tf.concat([z, h], axis=-1)

        # 全結合層
        x = self.fc(x)

        # 行動ごとに異なる活性化関数を適用
        steering = tf.tanh(x[:, 0:1])
        gas = tf.sigmoid(x[:, 1:2])
        brake = tf.sigmoid(x[:, 2:3])

        # 3つの行動を結合
        action = tf.concat(
            [steering, gas, brake],
            axis=-1
        )

        return action
