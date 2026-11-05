
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from VAE import _VAE


# 学習済みVAEを生成
vae = _VAE()

# 重みを初期化
dummy = tf.zeros((1, 64, 64, 3))
vae(dummy)

# 学習済み重みを読み込む
vae.load_weights("models/vae.weights.h5")

# 保存した画像を読み込む
with np.load("data/carRacing/episode_0000.npz") as data:
    images = data["images"]


# 複数の時刻を選択
indices = np.linspace(
    0,
    len(images) - 1,
    5,
    dtype=int
)

# 元画像
x = images[indices].astype(np.float32) / 255.0

# VAEで再構成
z = vae.encode(x, training=False)
reconstructed = vae.decode(z, training=False).numpy()

# 比較
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(5):
    axes[0, i].imshow(x[i])
    axes[0, i].axis("off")

    axes[1, i].imshow(reconstructed[i])
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Reconstructed")

plt.tight_layout()
plt.savefig("vae_comparison.png")
plt.show()
