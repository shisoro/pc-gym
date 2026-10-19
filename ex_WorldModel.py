import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as kl
import numpy as np
from dataclasses import dataclass

class _VAE(keras.Model):    # keras.Model : NNのlayerをまとめて、学習・推論・評価・保存などの機能を提供する中心的なクラス
    def __init__(self):     # self : 現在操作しているインスタンス自身を参照する変数
        super().__init__()

        self.z_size = 32 # 潜在空間のサイズ

        # --- encoder
        input_shape = (64, 64, 3)
        in_layer = c = kl.Input(shape=input_shape)  # 右辺の結果を両方の変数に代入, Input()の引数shapeに値を渡している。
        c = kl.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu")(c)
        c = kl.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")(c)
        c = kl.Conv2D(filters=128, kernel_size=4, strides=2, activation="relu")(c)
        c = kl.Conv2D(filters=256, kernel_size=4, strides=2, activation="relu")(c)
        c = kl.Flatten()(c) # 多次元のデータを１次元に変換する層を適用
        z_mean = kl.Dense(self.z_size)(c)   #
        z_log_var = kl.Dense(self.z_size)(c)
        self.encoder = keras.Model(in_layer, [z_mean, z_log_var])   # model = keras.Model(inputs, outputs)

        # --- decoder
        in_state = c = kl.Input(shape=(self.z_size,))
        c = kl.Dense(2 * 2* 256, activation="relu")(c)
        c = kl.Reshape((1, 1, 2 * 2 * 256))(c)
        c = kl.Conv2DTranspose(128, kernel_size=5, strides=2, padding="valid", activation="relu")(c)    # Conv2DTranspose : 転置畳み込み、空間サイズを大きくする
        c = kl.Conv2DTranspose(64, kernel_size=5, strides=2, padding="valid", activation="relu")(c)
        c = kl.Conv2DTranspose(32, kernel_size=6, strides=2, padding="valid", activation="relu")(c)
        c = kl.Conv2DTranspose(3, kernel_size=6, strides=2, padding="valid", activation="sigmoid")(c)   # sigmoid関数で出力値を0~1に収める
        self.decoder = keras.Model(in_state, c)

    def call(self, x):
        return self.decode(self.encode(x))

    def encode(self, x, training=False):
        z_mean, z_log_var = self.encoder(x, training=training)

        # reparameterize
        e = tf.random.normal(z_mean.shape)
        z = z_mean + tf.exp(0.5 * z_log_var) * e

        if training:
            return z_mean, z_log_var, z
        else:
            return z

    def decode(self, z, training=False):
        return self.decoder(z, training=training)

    def sample(self, size=1):
        z = np.random.normal(size=(size, self.z_size))
        return self.decode(z), z



class _MDNRNN(keras.Model):
    def __init__(self):
        super().__init__()

        self.action_num = 0    # アクション数
        self.z_size = 0        # VAEの潜在空間のサイズ
        self.num_mixture = 0   # 正規分布の数
        self.temperature = 0   # 温度パラメータ

        # --- RNN(LSTM)
        self.lstm_layer = kl.LSTM(256, return_sequences=True, return_state=True)

        # --- MDN
        self.mdn_layer = kl.Dense(self.z_size * self.num_mixture * 3)

    def call(self, z, onehot_actions, hidden_state, training=False):
        batch_size = z.shape[0] # zの行の数
        timesteps = z.shape[1]  # zの列の数

        # z + action
        x = tf.concat([z, onehot_actions], axis=2)

        # (batch, timesteps, z + action) -> (batch, timesteps, lstm_dim)
        x, h, c = self.lstm_layer(x, initial_state=hidden_state, training=training)

        # -> (batch * timesteps, lstm_dim)
        x = tf.reshape(x, (batch_size * timesteps, -1))

        # -> (batch * timesteps, z * num_mix * 3)
        x = self.mdn_layer(x, training=training)

        # -> (batch * timesteps, z, num_mix * 3)
        x = tf.reshape(x, (-1, self.z_size, self.num_mixture * 3))

        # -> (batch * timesteps, z, num_mix) * 3
        pi, mu, log_sigma = tf.split(x, 3, axis=2)

        return pi, mu, log_sigma, [h, c]

    def forward(self, z, action, hidden_state, rnn_only: bool = True):
        """ 1ステップ進める（主にhidden_state用）"""

        onehot_actions = tf.one_hot(np.array([action]), self.action_num, axis=1)

        # (batch, shape) -> (batch, 1, shape)
        z = z[:, np.newaxis, ...]
        onehot_actions = onehot_actions[:, np.newaxis, ...]

        pi, mu, log_sigma, hidden_state = self(z, onehot_actions, hidden_state)

        # --- hidden_state を進めるだけの場合はここまで
        if rnn_only:
            return hidden_state

        # --- 次の状態も予測する場合
        batch = z.shape[0]
        z_size = z.shape[2] # zのチャンネル数（RGB画像の場合: 3）

        sigma = np.exp(log_sigma)

        if self.temperature > 0:
            # softmax
            pi /= self.temperature # adjust temperatures
            pi = pi - tf.reduce_max(pi, axis=1, keepdims=True)  # overflow_protection
            exp_pi = tf.exp(pi)
            pi = exp_pi / tf.reduce_sum(exp_pi, axis=1, keepdims=True)

        samples = np.zeros((batch, z_size))
        for i in range(batch):
            for j in range(z_size):
                if self.temperature == 0:
                    # 最大値(決定的)
                    idx = np.argmax(pi[i][j])
                    z = mu[i][j][idx]
                else:
                    idx = random.choices([i for i in range(self.num_mixture)], weights=pi[i][j])
                    z = np.random.normal(mu[i][j][idx], sigma[i][j][idx] * self.temperature)
                samples[i][j] = z

        return samples, hidden_state

    def get_initial_state(self):
        return self.lstm_layer.cell.get_initial_state(batch_size=1, dtype=tf.float32)











# --------------------------------------------------
# 離散アクション用Configのベース（擬似コード）
class DiscreteActionConfig:
    action_num: int = 4 # 行動の選択肢の数（例: 上下左右の4方向）

@dataclass
class Config(DiscreteActionConfig):
    discount: float = 0.99  # 割引率
    lr: float = 0.001            # 学習率
    batch_size: int = 64         # バッチサイズ
    epsilon: float = 0.1         # ε-greedy法の探索率



class Config(DiscreteActionConfig):

    train_mode: int = 1  # 学習項目変更用です

    lr: float = 0.001
    batch_size: int = 32
    capacity: int = 100_000
    memory_warmup_size: int = 100

    # VAE
    z_size: int = 32
    kl_tolerance: float = 0.5

    # MDN-RNN
    sequence_length: int = 10
    rnn_units: int = 256
    num_mixture: int = 5  # number of mixtures in MDN
    temperature: float = 1.15

    # GA
    num_simulations: int = 16
    num_individual: int = 16
    mutation: float = 0.01
    randn_sigma: float = 1.0
    blx_a: float = 0.1