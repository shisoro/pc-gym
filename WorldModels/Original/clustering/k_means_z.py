
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples



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

    # 潜在表現の次元数の設定
    z_dim = 10
    
    # PCAに入力する元データを保存
    z_original = z.copy()

    # PCA
    pca = PCA(n_components=z_dim)
    z_pca = pca.fit_transform(z_original)

    # クラスタ数の最大値を設定
    n_cluster = 32

    sse = []
    mean_scores = []
    min_scores = []
    negative_ratios = []

    for k in range(2, n_cluster+1):

        # KMeans
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        kmeans.fit(z_pca)
        sse.append(kmeans.inertia_)

        labels = kmeans.fit_predict(z_pca)

        # Silhouette Score
        scores = silhouette_samples(z_pca, labels)

        # mean Silhouette Score
        mean_score = np.mean(scores)

        # min Silhouette Score
        min_score = np.min(scores)

        # negative Silhouette Score ratio
        negative_ratio = np.mean(scores < 0)

        mean_scores.append(mean_score)
        min_scores.append(min_score)
        negative_ratios.append(negative_ratio)

        print(f"クラスタ数：{k} Mean Silhouette Score: {mean_score}")
        print(f"クラスタ数：{k} Min Silhouette Score: {min_score}")
        print(f"クラスタ数：{k} Negative Ratio: {negative_ratio}")

    # エルボー法
    fig = plt.figure()
    ax = fig.subplots()

    x = list(range(2, n_cluster+1))
    y = np.array(sse)

    ax.plot(x,y)

    ax.set_title('Elbow Curve(z_dim = 10)')
    ax.set_xlabel('k')
    ax.set_ylabel('SSE')

    ax.legend()

    plt.show()

    # シルエット法
    fig = plt.figure() 
    ax = fig.subplots()

    x = list(range(2, n_cluster+1))
    y1 = np.array(mean_scores)
    y2 = np.array(min_scores)
    y3 = np.array(negative_ratios)

    # MeanSSのグラフ作成
    ax.plot(x, y1, label='blue', color='blue', linestyle='-', marker='o')

    # MinSSのグラフ作成
    ax.plot(x, y2, label='green', color='green', linestyle='-', marker='o')

    # NRのグラフ作成
    ax.plot(x, y3, label='red', color='red', linestyle='-', marker='o')

    ax.set_title('Silhouette Score(z_dim = 10)')
    ax.set_xlabel('k')
    ax.set_ylabel('Score')

    ax.legend()

    plt.show()


    # PCAで2次元に射影
    pca_2 = PCA(n_components=3)
    z_2 = pca_2.fit_transform(z)    

    # 2つの散布図を並べる
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pca_10 = PCA(n_components=10)
    z_10 = pca_10.fit_transform(z)

    k = 3

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