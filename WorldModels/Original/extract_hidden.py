
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import Config
from MDN_RNN import _MDNRNN


BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "data/rnn"
OUTPUT_DIR = BASE_DIR / "data/hidden"
WEIGHTS_PATH = BASE_DIR / "models/mdn_rnn.weights.h5"


def load_rnn(config):
    """
    学習済みMDN-RNNを読み込む。
    モデルの重みは更新しない。
    """
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"重みファイルが見つかりません: {WEIGHTS_PATH}"
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
        hidden_state=rnn.get_initial_state(),
        training=False
    )

    # 学習済み重みを読み込む
    rnn.load_weights(WEIGHTS_PATH)

    # 推論専用
    rnn.trainable = False

    return rnn


def extract_episode(rnn, config, input_path):
    """
    1エピソードから各時刻の隠れ状態を取得する。

    z       : (T+1, z_size)
    actions : (T, action_num)
    h       : (T+1, rnn_units)
    """

    with np.load(input_path, allow_pickle=False) as data:
        z = np.asarray(data["z"], dtype=np.float32)
        actions = np.asarray(
            data["actions"], dtype=np.float32
        )

    # データ形状の確認
    if z.ndim != 2 or z.shape[1] != config.z_size:
        raise ValueError(
            f"{input_path.name}: zの形状が不正です: {z.shape}"
        )

    if (
        actions.ndim != 2
        or actions.shape[1] != config.action_num
    ):
        raise ValueError(
            f"{input_path.name}: actionsの形状が不正です: "
            f"{actions.shape}"
        )

    if len(z) != len(actions) + 1:
        raise ValueError(
            f"{input_path.name}: 時刻数が一致しません。"
            f"z={len(z)}, actions={len(actions)}"
        )

    if len(actions) == 0:
        raise ValueError(
            f"{input_path.name}: 行動データが空です"
        )

    if not np.isfinite(z).all():
        raise ValueError(
            f"{input_path.name}: zにNaNまたはInfがあります"
        )

    if not np.isfinite(actions).all():
        raise ValueError(
            f"{input_path.name}: actionsにNaNまたはInfがあります"
        )

    # バッチ次元を追加
    # (T, D) -> (1, T, D)
    z_input = tf.convert_to_tensor(
        z[:-1][None, :, :],
        dtype=tf.float32
    )

    action_input = tf.convert_to_tensor(
        actions[None, :, :],
        dtype=tf.float32
    )

    # MDN-RNNのcall()と同じ入力を構築
    x = tf.concat(
        [z_input, action_input],
        axis=-1
    )

    # エピソードごとに初期状態をリセット
    initial_state = rnn.get_initial_state(batch_size=1)

    # 学習済みLSTMを推論モードで実行
    h_sequence, h_final, c_final = rnn.lstm_layer(
        x,
        initial_state=initial_state,
        training=False
    )

    # (1, T, rnn_units) -> (T, rnn_units)
    h_after = h_sequence.numpy()[0]

    # h_0 = 0 を先頭に追加
    h_initial = initial_state[0].numpy()

    h = np.concatenate(
        [h_initial, h_after],
        axis=0
    )

    if h.shape != (len(z), rnn.lstm_layer.units):
        raise ValueError(
            f"{input_path.name}: hの形状が不正です: {h.shape}"
        )

    if not np.isfinite(h).all():
        raise ValueError(
            f"{input_path.name}: hにNaNまたはInfがあります"
        )

    # 最終時刻の隠れ状態が系列末尾と一致するか確認
    np.testing.assert_allclose(
        h[-1],
        h_final.numpy()[0],
        rtol=1e-5,
        atol=1e-6
    )

    return z, actions, h


def main():
    config = Config()

    files = sorted(INPUT_DIR.glob("episode_*.npz"))

    if not files:
        raise FileNotFoundError(
            f"エピソードが見つかりません: {INPUT_DIR}"
        )

    rnn = load_rnn(config)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for input_path in files:
        output_path = OUTPUT_DIR / input_path.name

        # 既存ファイルの誤上書きを防止
        if output_path.exists():
            raise FileExistsError(
                f"出力ファイルが既に存在します: {output_path}"
            )

        z, actions, h = extract_episode(
            rnn,
            config,
            input_path
        )

        # 新しいファイルとして保存
        np.savez_compressed(
            output_path,
            z=z,
            actions=actions,
            h=h
        )

        print(
            f"{input_path.name}: "
            f"z={z.shape}, "
            f"actions={actions.shape}, "
            f"h={h.shape}"
        )

    print("隠れ状態の抽出が完了しました。")


if __name__ == "__main__":
    main()
