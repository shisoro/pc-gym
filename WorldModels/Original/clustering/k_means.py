
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



def main():

    # 潜在変数zのファイル一覧
    files = sorted(
        glob.glob("../data/rnn/episode_*.npz")
    )

    if not files:
        raise FileNotFoundError(
            "潜在変数zのデータが見つかりません"
        )

    all_z = []

    for episode_id, file in enumerate(files):

        with np.load(file) as data:
            z = data["z"]   # z_mean

        all_z.append(z)

    z = np.concatenate(all_z, axis=0)

    # クラスタ数を設定
    k = 3
    
    # PCAに入力する元データを保存
    z_original = z.copy()

    scores = []

    for i in range(32):
        # PCA
        pca = PCA(n_components=i+1)
        z_pca = pca.fit_transform(z_original)

        # KMeans
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(z_pca)

        # Silhouette Score
        score = silhouette_score(z_pca, labels)
        scores.append(score)

        print(f"{i+1}次元 Silhouette Score: {score}")



    # PCAで2次元に射影
    pca_2 = PCA(n_components=3)
    z_2 = pca_2.fit_transform(z)    

    # 2つの散布図を並べる
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pca_10 = PCA(n_components=10)
    z_10 = pca_10.fit_transform(z)


    # KMeans
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels_32 = kmeans.fit_predict(z)

    labels_10 = kmeans.fit_predict(z_10)

    

    # 32次元でK-meansを行った結果
    axes[0].scatter(
        z_2[:, 0],
        z_2[:, 1],
        c=labels_32,
        cmap="tab10",
        s=5,
        alpha=0.6
    )

    axes[0].set_title("K-means (32 dimensions)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    # 10次元でK-meansを行った結果
    axes[1].scatter(
        z_2[:, 0],
        z_2[:, 1],
        c=labels_10,
        cmap="tab10",
        s=5,
        alpha=0.6
    )

    axes[1].set_title("K-means (10 dimensions)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()   