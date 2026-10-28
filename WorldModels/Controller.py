import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as kl

class _Controller(keras.Model):
    def __init__(self, config):
        super().__init__()

        self.z_size = config.z_size
        self.hidden_size = config.hidden_size
        self.action_num = config.action_num

        # Controller
        self.fc = kl.Dense(self.action_num)

    def call(self, z, hidden_state):

        # 潜在変数と隠れ状態を結合
        x = tf.concat(
            [z, hidden_state],
            axis=-1
        )

        # 線形変換
        q = self.fc(x)

        return q