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
    def __init__(self, config):
        super().__init__()
        # ()内の数字は例
        self.action_num = config.action_num    # アクション数(3)
        self.z_size = config.z_size            # VAEの潜在空間のサイズ(32)
        self.num_mixture = config.num_mixture   # 正規分布の数(5)
        self.temperature = config.temperature   # 温度パラメータ(1.15)

        # --- RNN(LSTM) --- 
        ## x, h, c = kl.LSTM(隠れ状態の次元数, すべての時刻の隠れ状態を返すかどうか, 最後の時刻の隠れ状態を返すかどうか)
        self.lstm_layer = kl.LSTM(
            config.rnn_units, # (256)
            return_sequences=True, 
            return_state=True
        )

        # --- MDN --- 
        ## 全結合層の計算: y = Wx+b -> MDN
        self.mdn_layer = kl.Dense(
            self.z_size * self.num_mixture * 3
        )

    def call(self, z, onehot_actions, hidden_state, training=False):
        # z + action  (32,3)   ## 入力と行動を統合して一つの入力データにする
        x = tf.concat([z, onehot_actions], axis=-1)     # axis=-1は最後の軸

        # LSTMの実行    LSTMの出力：(32,3,256)
        x, h, c = self.lstm_layer(
            x, 
            initial_state=hidden_state,
            training=training
        )

        # MDNによる確率分布の生成   MDNの出力：(32,3,480)
        x = self.mdn_layer(x, training=training)
            
        ## 混合係数,平均,標準偏差を渡しやすくするために次元を調整         
        shape = tf.shape(x) # (32,3,480)
        # (batch, timesteps, z_size, num_mix * 3) = (32, 3, 32, 15)
        x = tf.reshape(x, (shape[0], shape[1], self.z_size, self.num_mixture * 3))

        # 混合分布のパラメータ
        # 15種類のパラメータを３つに分割
        pi, mu, log_sigma = tf.split(x, 3, axis=-1)  ## 各値(32,3,32,5)を取得する 

        return pi, mu, log_sigma, [h, c]

    def forward(self, z, action, hidden_state, rnn_only: bool = True):
        # z: (batch, z_size)
        # action: (batch,)の整数行動

        # リストやnumpy配列 -> テンソル
        z = tf.convert_to_tensor(z, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)

        # 時系列次元を追加
        z = tf.expand_dims(z, axis=1)

        # One-hotに変換して時系列次元を追加
        onehot_actions = tf.one_hot(
            action, 
            depth=self.action_num, 
        )
        onehot_actions = tf.expand_dims(
            onehot_actions, axis=1
        )

        pi, mu, log_sigma, hidden_state = self(
            z, 
            onehot_actions, 
            hidden_state,
            training=False
        )

        # --- hidden_state を進めるだけの場合はここまで
        if rnn_only:
            return hidden_state

        # 時系列次元を削除
        # (batch, 1, z_size, num_mixture)
        # -> (batch, z_size, num_mixture)
        pi = tf.squeeze(pi, axis=1)
        mu = tf.squeeze(mu, axis=1)
        log_sigma = tf.squeeze(log_sigma, axis=1)

        if self.temperature == 0:
            # 最も大きな混合係数の成分を選択
            idx = tf.argmax(pi, axis=-1, output_type=tf.int32)
        
        if self.temperature > 0:
            # softmax
            # 混合係数を温度調整
            logits = pi / self.temperature

            # カテゴリ分布から混合成分を選択
            shape = tf.shape(logits)      
            flat_logits = tf.reshape(
                logits, (-1, self.num_mixture)
            )
            # 確率を決定する際にsoftmax関数を使用して正規化している
            # 入力が二次元テンソルなのは、バッチ処理による並列処理のため
            idx = tf.random.categorical(
                flat_logits, num_samples=1
            )

            idx = tf.reshape(
                idx, (shape[0], shape[1])
            )   

        # 選択した混合成分の平均と標準偏差
        selected_mu = tf.gather(
            mu, idx, axis=2, batch_dims=2
        )

        selected_log_sigma = tf.gather(
            log_sigma, idx, axis=2, batch_dims=2
        )
        
        if self.temperature ==0:
            samples = selected_mu   # ランダム性を持たせないために平均を返す
        else:
            # 選ばれた正規分布から値をサンプリングして、次の潜在変数とする
            noise = tf.random.normal(tf.shape(selected_mu))
            # 標準正規分布： X=μ+σε
            samples = (
                selected_mu
                + tf.exp(selected_log_sigma)
                * self.temperature
                * noise
            )

        return samples, hidden_state

    # --- LSTMの初期状態を生成
    def get_initial_state(self):
        return self.lstm_layer.cell.get_initial_state(batch_size=1, dtype=tf.float32)



class _Controller(keras.Model):
    def __init__(self):
        super().__init__()
        self.out_layer = kl.Dense(self.action_num)  # アクション数が引数

    def call(self, z, hidden_state):
        # a_t = W_c[z_t h_t] + b_c  # 累計報酬を最大化するように予想された方策関数
        x = tf.concat([z, hidden_state[1]], axis=1)
        return self.out_layer(x)








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