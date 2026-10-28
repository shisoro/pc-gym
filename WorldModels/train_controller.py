import numpy as np
import cma

def get_flat_params(model):
    """Controllerの全パラメータを1次元に変換"""

    weights = model.get_weights()

    flat_params = np.concatenate([
        w.flatten() for w in weights
    ])

    return flat_params

def set_flat_params(model, flat_params):
    """1次元パラメータをControllerに設定"""

    weights = model.get_weights()

    new_weights = []
    index = 0

    for w in weights:
        size = w.size

        new_w = flat_params[index:index + size]
        new_w = new_w.reshape(w.shape)

        new_weights.append(new_w)
        index += size

    if index != len(flat_params):
        raise ValueError("パラメータ数が一致しません")

    model.set_weights(new_weights)

def train_controller(
    controller,
    evaluate_fn,
    generations=100,
    population_size=16,
    sigma=0.1
):
    # Controllerの初期パラメータ
    initial_params = get_flat_params(controller)

    # CMA-ESを初期化
    es = cma.CMAEvolutionStrategy(
        initial_params,
        sigma,
        {"popsize": population_size}
    )

    best_reward = -np.inf
    best_params = initial_params.copy()

    for generation in range(generations):

        if es.stop():
            break

        # 個体の生成
        solutions = es.ask()

        fitness_values = []

        for params in solutions:

            # Controllerの重みを更新
            set_flat_params(controller, params)

            # 環境で評価
            reward = evaluate_fn(controller)

            # 最大化問題を最小化問題へ
            fitness = -reward

            fitness_values.append(fitness)

            if reward > best_reward:
                best_reward = reward
                best_params = params.copy()

        # CMA-ESの分布を更新
        es.tell(solutions, fitness_values)

        print(
            f"Generation: {generation}, "
            f"Best reward: {best_reward:.4f}"
        )

    # 最良のControllerを復元
    set_flat_params(controller, best_params)

    return best_params, best_reward