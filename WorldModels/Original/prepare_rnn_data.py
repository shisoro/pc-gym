
import glob
import os

import numpy as np
import tensorflow as tf

from VAE import _VAE


def main():

    # 学習済みVAEを読み込む
    vae = _VAE()

    dummy = tf.zeros((1, 64, 64, 3))
    vae(dummy, training=False)

    vae.load_weights("models/vae.weights.h5")

    # 元データのファイル一覧
    files = sorted(
        glob.glob("data/carRacing/episode_*.npz")
    )

    if not files:
        raise FileNotFoundError(
            "CarRacingのデータが見つかりません"
        )

    os.makedirs("data/rnn", exist_ok=True)

    for episode_id, file in enumerate(files):

        with np.load(file) as data:
            images = data["images"]
            actions = data["actions"]

        # 画素値を0〜1へ正規化
        images = images.astype(np.float32) / 255.0

        # VAEのEncoderで潜在変数を生成（vae.encode() -> z=μ+σε, Model.vae.encoder() -> z=μ(x)）
        # サンプリングによるばらつきを避けるため平均を使用（学習を簡単にするため）
        z_mean, z_log_var = vae.encoder(
            images,
            training=False
        )

        z = z_mean.numpy()

        # データの時間的な対応関係を確認
        assert len(z) == len(actions) + 1

        # エピソード単位で保存
        output_path = (
            f"data/rnn/episode_{episode_id:04d}.npz"
        )

        np.savez_compressed(
            output_path,
            z=z,
            actions=actions
        )

        print(
            f"Episode {episode_id}: "
            f"z={z.shape}, "
            f"actions={actions.shape}"
        )


if __name__ == "__main__":
    main()
