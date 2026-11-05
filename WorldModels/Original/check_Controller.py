import tensorflow as tf

from config import Config
from Controller import _Controller

config = Config()

controller = _Controller(config)

# ダミーの潜在変数
z = tf.random.normal((32, config.z_size))

# ダミーのLSTM状態
h = tf.random.normal((32, config.rnn_units))
c = tf.random.normal((32, config.rnn_units))

hidden_state = [h, c]

# Controllerの実行
action = controller(z, hidden_state)

print("action shape:", action.shape)
print("steering range:", tf.reduce_min(action[:, 0]).numpy(),
      tf.reduce_max(action[:, 0]).numpy())
print("gas range:", tf.reduce_min(action[:, 1]).numpy(),
      tf.reduce_max(action[:, 1]).numpy())
print("brake range:", tf.reduce_min(action[:, 2]).numpy(),
      tf.reduce_max(action[:, 2]).numpy())