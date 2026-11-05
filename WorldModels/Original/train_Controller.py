import os
import numpy as np
import tensorflow as tf
import gymnasium as gym
import cma

from config import Config
from VAE import _VAE
from MDN_RNN import _MDNRNN
from Controller import _Controller
from evaluate import evaluate_controller


# -----------------------------
# モデルの初期化と重みの読み込み
# -----------------------------
def load_models(config):

    # VAE
    vae = _VAE()

    dummy_image = tf.zeros((1, 64, 64, 3))
    vae(dummy_image, training=False)

    vae.load_weights("models/vae.weights.h5")

    # MDN-RNN
    rnn = _MDNRNN(config)

    dummy_z = tf.zeros((1, 1, config.z_size))
    dummy_action = tf.zeros((1, 1, config.action_num))

    initial_state = rnn.get_initial_state(batch_size=1)

    rnn(
        dummy_z,
        dummy_action,
        initial_state,
        training=False
    )

    rnn.load_weights("models/mdn_rnn.weights.h5")

    # Controller
    controller = _Controller(config)

    dummy_hidden = rnn.get_initial_state(batch_size=1)

    controller(
        tf.zeros((1, config.z_size)),
        dummy_hidden
    )

    return vae, rnn, controller


# -----------------------------
# Controllerの重みを1次元配列に変換
# -----------------------------
def get_controller_params(controller):

    weights = controller.fc.get_weights()

    kernel = weights[0]
    bias = weights[1]

    params = np.concatenate([
        kernel.flatten(),
        bias.flatten()
    ])

    return params


# -----------------------------
# 1次元配列から重みを復元
# -----------------------------
def set_controller_params(controller, params):

    kernel_shape = controller.fc.get_weights()[0].shape
    bias_shape = controller.fc.get_weights()[1].shape

    kernel_size = np.prod(kernel_shape)
    bias_size = np.prod(bias_shape)

    kernel = params[:kernel_size].reshape(kernel_shape)

    bias = params[
        kernel_size:kernel_size + bias_size
    ].reshape(bias_shape)

    controller.fc.set_weights([kernel, bias])


# -----------------------------
# CMA-ESによる学習
# -----------------------------
def train_controller():

    config = Config()

    # モデル読み込み
    vae, rnn, controller = load_models(config)

    # 環境作成
    env = gym.make("CarRacing-v3")

    # 初期パラメータ
    initial_params = get_controller_params(controller)

    # CMA-ESの初期化
    es = cma.CMAEvolutionStrategy(
        initial_params,
        config.cma_sigma,
        {
            "popsize": config.num_individual,
            "maxiter": config.num_generations,
            "verb_disp": 1
        }
    )

    generation = 0

    try:
        while not es.stop():

            generation += 1

            # 個体の生成
            solutions = es.ask()

            fitnesses = []

            for i, solution in enumerate(solutions):

                # Controllerの重みを候補値に変更
                set_controller_params(
                    controller,
                    solution
                )

                # CarRacingで評価
                reward = evaluate_controller(
                    env,
                    vae,
                    rnn,
                    controller
                )

                # CMA-ESは最小化するため符号を反転
                fitness = -reward

                fitnesses.append(fitness)

                print(
                    f"Generation: {generation}, "
                    f"Individual: {i + 1}, "
                    f"Reward: {reward:.2f}"
                )

            # 集団の評価結果をCMA-ESに渡す
            es.tell(solutions, fitnesses)

            print(
                f"Generation {generation} completed. "
                f"Best reward: {-es.result.fbest:.2f}"
            )

    finally:
        env.close()

    # 最良パラメータを取得
    best_params = es.result.xbest

    # Controllerに反映
    set_controller_params(
        controller,
        best_params
    )

    # 保存
    os.makedirs("models", exist_ok=True)

    controller.save_weights(
        "models/controller.weights.h5"
    )

    print("Controllerの学習が完了しました。")


if __name__ == "__main__":
    train_controller()