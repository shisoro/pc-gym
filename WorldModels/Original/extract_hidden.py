
"""
学習済みMDN-RNNから隠れ状態を抽出する。

実行例:
    uv run python extract_hidden.py --mode dataset
    uv run python extract_hidden.py --mode controller --episodes 10
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import Config
from MDN_RNN import _MDNRNN


BASE_DIR = Path(__file__).resolve().parent


# ----------------------------------------
# MDN-RNNの読み込み
# ----------------------------------------
def build_rnn(config):

    path = BASE_DIR / "models/mdn_rnn.weights.h5"

    if not path.is_file():
        raise FileNotFoundError(
            f"MDN-RNNの重みが見つかりません: {path}"
        )

    rnn = _MDNRNN(config)

    # 重みを読み込む前にモデルを構築
    dummy_z = tf.zeros(
        (1, 1, config.z_size),
        dtype=tf.float32
    )

    dummy_action = tf.zeros(
        (1, 1, config.action_num),
        dtype=tf.float32
    )

    rnn(
        dummy_z,
        dummy_action,
        hidden_state=rnn.get_initial_state(batch_size=1),
        training=False
    )

    # 学習済み重みを読み込む
    rnn.load_weights(path)

    # 重みを更新しない
    rnn.trainable = False

    return rnn


# ----------------------------------------
# VAEの読み込み
# ----------------------------------------
def build_vae():

    from VAE import _VAE

    path = BASE_DIR / "models/vae.weights.h5"

    if not path.is_file():
        raise FileNotFoundError(
            f"VAEの重みが見つかりません: {path}"
        )

    vae = _VAE()

    # モデルを構築
    vae(
        tf.zeros(
            (1, 64, 64, 3),
            dtype=tf.float32
        )
    )

    vae.load_weights(path)
    vae.trainable = False

    return vae


# ----------------------------------------
# Controllerの読み込み
# ----------------------------------------
def build_controller(config, rnn):

    from Controller import _Controller

    path = BASE_DIR / "models/controller_40_gen.weights.h5"

    if not path.is_file():
        raise FileNotFoundError(
            f"Controllerの重みが見つかりません: {path}"
        )

    controller = _Controller(config)

    # モデルを構築
    controller(
        tf.zeros(
            (1, config.z_size),
            dtype=tf.float32
        ),
        rnn.get_initial_state(batch_size=1)
    )

    controller.load_weights(path)
    controller.trainable = False

    return controller


# ----------------------------------------
# 数値の検証
# ----------------------------------------
def ensure_finite(name, array):

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name}にNaNまたはInfが含まれています"
        )


def check_arrays(z, actions, h, config):

    if z.ndim != 2 or z.shape[1] != config.z_size:
        raise ValueError(
            f"zの形状が不正です: {z.shape}"
        )

    if (
        actions.ndim != 2
        or actions.shape[1] != config.action_num
    ):
        raise ValueError(
            f"actionsの形状が不正です: {actions.shape}"
        )

    if (
        h.ndim != 2
        or h.shape[0] != z.shape[0]
        or h.shape[1] != config.rnn_units
    ):
        raise ValueError(
            f"hの形状が不正です: {h.shape}"
        )

    if len(z) != len(actions) + 1:
        raise ValueError(
            "時刻数が一致しません: "
            f"z={len(z)}, actions={len(actions)}"
        )

    for name, array in (
        ("z", z),
        ("actions", actions),
        ("h", h)
    ):
        ensure_finite(name, array)


# ----------------------------------------
# 既存ファイルを上書きせず保存
# ----------------------------------------
def save_npz_exclusive(path, **arrays):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # xb: ファイルが既に存在する場合はエラー
    with path.open("xb") as file:
        try:
            np.savez_compressed(
                file,
                **arrays
            )
        except BaseException:
            file.close()
            path.unlink(missing_ok=True)
            raise


# ----------------------------------------
# 保存済みエピソードからhを抽出
# ----------------------------------------
def extract_dataset_episode(rnn, config, path):

    with np.load(
        path,
        allow_pickle=False
    ) as data:

        z = np.asarray(
            data["z"],
            dtype=np.float32
        )

        actions = np.asarray(
            data["actions"],
            dtype=np.float32
        )

    # データの形状を確認
    if z.ndim != 2 or z.shape[1] != config.z_size:
        raise ValueError(
            f"{path.name}: zの形状が不正です: {z.shape}"
        )

    if (
        actions.ndim != 2
        or actions.shape[1] != config.action_num
    ):
        raise ValueError(
            f"{path.name}: actionsの形状が不正です: "
            f"{actions.shape}"
        )

    if len(z) != len(actions) + 1:
        raise ValueError(
            f"{path.name}: zとactionsの時刻数が一致しません"
        )

    if len(actions) == 0:
        raise ValueError(
            f"{path.name}: actionsが空です"
        )

    ensure_finite("z", z)
    ensure_finite("actions", actions)

    # z: (T+1, z_size)
    # actions: (T, action_num)
    #
    # RNNへ入力するのはz_0〜z_(T-1)
    z_input = tf.convert_to_tensor(
        z[:-1][None, :, :],
        dtype=tf.float32
    )

    action_input = tf.convert_to_tensor(
        actions[None, :, :],
        dtype=tf.float32
    )

    # MDN-RNNの入力を作成
    x = tf.concat(
        [z_input, action_input],
        axis=-1
    )

    # エピソードごとにLSTMを初期化
    initial_state = rnn.get_initial_state(
        batch_size=1
    )

    # 全時刻の隠れ状態を取得
    h_sequence, h_final, c_final = rnn.lstm_layer(
        x,
        initial_state=initial_state,
        training=False
    )

    # (1, T, rnn_units) -> (T, rnn_units)
    h_after = h_sequence.numpy()[0]

    # 初期状態h_0を先頭に追加
    h_initial = initial_state[0].numpy()

    h = np.concatenate(
        [h_initial, h_after],
        axis=0
    )

    # 最終状態の整合性を確認
    np.testing.assert_allclose(
        h[-1],
        h_final.numpy()[0],
        rtol=1e-5,
        atol=1e-6
    )

    check_arrays(z, actions, h, config)

    return z, actions, h


# ----------------------------------------
# 観測画像からzを取得
# ----------------------------------------
def encode_observation(vae, obs):

    # evaluate.pyと同じ前処理
    x = tf.convert_to_tensor(obs)

    x = tf.image.resize(
        x,
        (64, 64)
    )

    x = tf.cast(
        x,
        tf.float32
    ) / 255.0

    x = tf.expand_dims(
        x,
        axis=0
    )

    # VAEの平均を潜在変数として使用
    z_mean, z_log_var = vae.encoder(
        x,
        training=False
    )

    return tf.cast(
        z_mean,
        tf.float32
    )


# ----------------------------------------
# Controllerの方策でエピソードを収集
# ----------------------------------------
def collect_controller_episode(
    env,
    vae,
    rnn,
    controller,
    config,
    max_steps,
    seed
):

    obs, info = env.reset(seed=seed)

    # LSTM初期状態
    hidden_state = rnn.get_initial_state(
        batch_size=1
    )

    z_values = []
    h_values = []
    actions = []
    rewards = []

    terminated = False
    truncated = False

    for step in range(max_steps):

        # 現在の観測から潜在変数を取得
        z_tensor = encode_observation(
            vae,
            obs
        )

        # 現在の隠れ状態を使って行動を決定
        action_tensor = controller(
            z_tensor,
            hidden_state
        )

        action = (
            action_tensor.numpy()[0]
            .astype(np.float32)
        )

        z = (
            z_tensor.numpy()[0]
            .astype(np.float32)
        )

        # 行動決定前のh_t
        h = (
            hidden_state[0].numpy()[0]
            .astype(np.float32)
        )

        # 保存前に数値を確認
        for name, array in (
            ("z", z),
            ("h", h),
            ("action", action)
        ):
            ensure_finite(name, array)

        # evaluate_controller()と同じ順序で
        # RNNの隠れ状態を更新
        next_hidden_state = rnn.rnn(
            z_tensor,
            action.reshape(
                1,
                config.action_num
            ),
            hidden_state=hidden_state,
            rnn_only=True
        )

        # 環境を進める
        next_obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        # 各時刻のデータを保存
        z_values.append(z.copy())
        h_values.append(h.copy())
        actions.append(action.copy())
        rewards.append(float(reward))

        # 次時刻へ
        hidden_state = next_hidden_state
        obs = next_obs

        if terminated or truncated:
            break

    if not actions:
        raise ValueError(
            "行動データを取得できませんでした"
        )

    # 最終観測z_Tを追加
    z_final = encode_observation(
        vae,
        obs
    ).numpy()[0].astype(np.float32)

    z_values.append(z_final)

    # 最終隠れ状態h_Tを追加
    h_final = (
        hidden_state[0].numpy()[0]
        .astype(np.float32)
    )

    h_values.append(h_final)

    # NumPy配列へ変換
    z = np.stack(z_values)
    h = np.stack(h_values)
    actions = np.stack(actions)

    rewards = np.asarray(
        rewards,
        dtype=np.float32
    )

    check_arrays(
        z,
        actions,
        h,
        config
    )

    ensure_finite(
        "rewards",
        rewards
    )

    return (
        z,
        actions,
        h,
        rewards,
        terminated,
        truncated
    )


# ----------------------------------------
# datasetモード
# ----------------------------------------
def run_dataset(config, rnn, args):

    input_dir = BASE_DIR / "data/rnn"
    output_dir = BASE_DIR / "data/hidden"

    files = sorted(
        input_dir.glob("episode_*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"エピソードが見つかりません: {input_dir}"
        )

    for path in files:

        output_path = (
            output_dir / path.name
        )

        if output_path.exists():
            raise FileExistsError(
                f"出力ファイルが既に存在します: {output_path}"
            )

        z, actions, h = extract_dataset_episode(
            rnn,
            config,
            path
        )

        save_npz_exclusive(
            output_path,
            z=z,
            actions=actions,
            h=h
        )

        print(
            f"{path.name}: "
            f"z={z.shape}, "
            f"actions={actions.shape}, "
            f"h={h.shape}"
        )


# ----------------------------------------
# controllerモード
# ----------------------------------------
def run_controller(config, rnn, args):

    import gymnasium as gym

    vae = build_vae()

    controller = build_controller(
        config,
        rnn
    )

    output_dir = (
        BASE_DIR / "data/controller_hidden"
    )

    # 既存ファイルの上書きを防止
    for episode_id in range(args.episodes):

        path = (
            output_dir
            / f"episode_{episode_id:04d}.npz"
        )

        if path.exists():
            raise FileExistsError(
                f"出力ファイルが既に存在します: {path}"
            )

    # CarRacing環境を作成
    env = gym.make(
        "CarRacing-v3",
        render_mode="human" if args.render else None
    )

    try:
        for episode_id in range(args.episodes):

            seed = (
                None
                if args.seed is None
                else args.seed + episode_id
            )

            (
                z,
                actions,
                h,
                rewards,
                terminated,
                truncated
            ) = collect_controller_episode(
                env,
                vae,
                rnn,
                controller,
                config,
                args.max_steps,
                seed
            )

            output_path = (
                output_dir
                / f"episode_{episode_id:04d}.npz"
            )

            save_npz_exclusive(
                output_path,
                z=z,
                actions=actions,
                h=h,
                rewards=rewards,
                total_reward=np.float32(
                    rewards.sum()
                ),
                terminated=np.bool_(
                    terminated
                ),
                truncated=np.bool_(
                    truncated
                ),
                seed=np.int64(seed)
            )

            print(
                f"{output_path.name}: "
                f"z={z.shape}, "
                f"actions={actions.shape}, "
                f"h={h.shape}, "
                f"reward={rewards.sum():.2f}"
            )

    finally:
        env.close()


# ----------------------------------------
# メイン処理
# ----------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="MDN-RNNの隠れ状態を抽出する"
    )

    parser.add_argument(
        "--mode",
        choices=("dataset", "controller"),
        required=True
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--render",
        action="store_true"
    )

    args = parser.parse_args()

    if args.episodes < 1 or args.max_steps < 1:
        parser.error(
            "--episodesと--max-stepsは1以上にしてください"
        )

    config = Config()

    # MDN-RNNを読み込む
    rnn = build_rnn(config)

    if args.mode == "dataset":
        run_dataset(
            config,
            rnn,
            args
        )

    elif args.mode == "controller":
        run_controller(
            config,
            rnn,
            args
        )


if __name__ == "__main__":
    main()
