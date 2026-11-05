
import os
import glob
import numpy as np
import tensorflow as tf

from tensorflow import keras

from config import Config
from VAE import _VAE
from train_VAE import train_vae_step


def load_vae_data(data_dir="data/carRacing"):

    files = sorted(
        glob.glob(f"{data_dir}/episode_*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"データが見つかりません: {data_dir}"
        )

    image_list = []

    for file in files:
        with np.load(file) as data:
            image_list.append(data["images"])

    images = np.concatenate(
        image_list, axis=0
    )

    images = images.astype(np.float32) / 255.0

    return images


def main():

    config = Config()

    # 画像データを読み込む
    images = load_vae_data()

    print("Images:", images.shape)

    # tf.data.Datasetを作成
    dataset = tf.data.Dataset.from_tensor_slices(
        images
    )

    dataset = dataset.shuffle(
        buffer_size=min(len(images), 10000)
    )

    dataset = dataset.batch(
        config.batch_size
    )

    # GPUが計算している間にBatchデータをCPU側に用意しておく機能
    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    # VAEを生成
    vae = _VAE()

    # ダミー画像を生成（Kerasにvaeモデルの存在を認識してもらうため）
    dummy = tf.zeros((1, 64, 64, 3))
    
    vae(dummy, training=False)

    print("VAE built:", vae.built)
    

    # 最適化アルゴリズム
    optimizer = keras.optimizers.Adam(
        learning_rate=config.learning_rate
    )

    epochs = 50

    for epoch in range(epochs):

        total_loss = 0.0
        total_rc = 0.0
        total_kl = 0.0
        steps = 0

        for x in dataset:

            loss, rc_loss, kl_loss = train_vae_step(
                vae, optimizer, x
            )

            total_loss += float(loss.numpy())
            total_rc += float(rc_loss.numpy())
            total_kl += float(kl_loss.numpy())

            steps += 1

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss={total_loss / steps:.4f} "
            f"RC={total_rc / steps:.4f} "
            f"KL={total_kl / steps:.4f}"
        )

    # 重みを保存
    os.makedirs("models", exist_ok=True)
    
    vae.save_weights(
        "models/vae.weights.h5"
    )

    print("VAEの学習が完了しました")


if __name__ == "__main__":
    main()
