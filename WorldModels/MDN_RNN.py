import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as kl

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
            self.z_size * self.num_mixture * 3  # 出力数
        )

    def call(self, z, onehot_actions, hidden_state, training=False):
        # z + action  (32,10,32) + (32,10,3) = (32,10,35)   ## 入力と行動を統合して一つの入力データにする
        x = tf.concat([z, onehot_actions], axis=-1)     # axis=-1は最後の軸

        # LSTMの実行    LSTMの出力：(32,10,256)
        x, h, c = self.lstm_layer(
            x, 
            initial_state=hidden_state,
            training=training
        )

        # MDNによる確率分布の生成   MDNの出力：(32,10,480)
        x = self.mdn_layer(x, training=training)
            
        ## 混合係数,平均,標準偏差を渡しやすくするために次元を調整         
        shape = tf.shape(x) # (32,10,480)
        # (batch, timesteps, z_size, num_mix * 3) = (32, 10, 32, 15)
        x = tf.reshape(x, (shape[0], shape[1], self.z_size, self.num_mixture * 3))

        # 混合分布のパラメータ
        # 15種類のパラメータを３つに分割
        pi, mu, log_sigma = tf.split(x, 3, axis=-1)  ## 各値(32,10,32,5)を取得する 

        return pi, mu, log_sigma, [h, c]

    def forward(self, z, action, hidden_state, rnn_only: bool = True):
        # z: (batch, z_size)    ## batch次元はVAEで付与
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
        # --- LSTMの初期状態を生成
    def get_initial_state(self, batch_size=1):
        h = tf.zeros(
            (batch_size, self.lstm_layer.units),
            dtype=tf.float32
        )

        c = tf.zeros(
            (batch_size, self.lstm_layer.units),
            dtype=tf.float32
        )

        return [h, c]
    