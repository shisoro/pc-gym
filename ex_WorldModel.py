import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as kl
import numpy as np

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

