import tensorflow as tf

from config import Config
from MDN_RNN import _MDNRNN
from train_RNN import train_rnn_step

config = Config()

rnn = _MDNRNN(config)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=config.learning_rate
)

z = tf.random.normal((32, 11, 32))
actions = tf.random.uniform((32, 10, 3))

loss = train_rnn_step(
    rnn,
    optimizer,
    z,
    actions
)

print("loss:", loss.numpy())
print("loss shape:", loss.shape)