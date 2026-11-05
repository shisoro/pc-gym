import gymnasium as gym
import tensorflow as tf

from config import Config
from VAE import _VAE
from MDN_RNN import _MDNRNN
from Controller import _Controller
from evaluate import evaluate_controller


config = Config()

# -------------------------
# VAEの読み込み
# -------------------------
vae = _VAE()
vae(tf.zeros((1, 64, 64, 3)))
vae.load_weights("models/vae.weights.h5")

# -------------------------
# MDN-RNNの読み込み
# -------------------------
rnn = _MDNRNN(config)

dummy_z = tf.zeros((1, 1, config.z_size))
dummy_action = tf.zeros((1, 1, config.action_num))

rnn(
    dummy_z,
    dummy_action,
    rnn.get_initial_state()
)

rnn.load_weights("models/mdn_rnn.weights.h5")

# -------------------------
# Controllerの読み込み
# -------------------------
controller = _Controller(config)

controller(
    tf.zeros((1, config.z_size)),
    rnn.get_initial_state()
)

controller.load_weights("models/controller.weights.h5")

# -------------------------
# 環境の作成と評価
# -------------------------
env = gym.make("CarRacing-v3", render_mode="human")

reward = evaluate_controller(
    env,
    vae,
    rnn,
    controller
)

print("Total reward:", reward)

env.close()