
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples



def main():

    # 隠れ状態hのファイル一覧
    files = sorted(
        glob.glob("../data/controller_hidden/episode_*.npz")
    )

    if not files:
        raise FileNotFoundError(
            "隠れ状態hのデータが見つかりません"
        )

    all_h = []

    for episode_id, file in enumerate(files):

        with np.load(file) as data:
            h = data["h"]

        all_h.append(h)

    h = np.concatenate(all_h, axis=0)

    # 隠れ状態の次元数の設定
    h_dim = 15
    
    # PCAに入力する元データを保存
    h_original = h.copy()

    # PCA
    pca = PCA(n_components=h_dim)
    h_pca = pca.fit_transform(h_original)

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

        kmeans.fit(h_pca)
        sse.append(kmeans.inertia_)

        labels = kmeans.fit_predict(h_pca)

        # Silhouette Score
        scores = silhouette_samples(h_pca, labels)

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

    ax.set_title('Elbow Curve(h_dim = 15)')
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

    ax.set_title('Silhouette Score(h_dim = 15)')
    ax.set_xlabel('k')
    ax.set_ylabel('Score')

    ax.legend()

    plt.show()


    # PCAで2次元に射影
    pca_2 = PCA(n_components=3)
    h_2 = pca_2.fit_transform(h)    

    # 2つの散布図を並べる
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pca_10 = PCA(n_components=10)
    h_10 = pca_10.fit_transform(h)

    k = 10

    # KMeans
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels_32 = kmeans.fit_predict(h)

    labels_10 = kmeans.fit_predict(h_10)

    

    # 32次元でK-meansを行った結果
    axes[0].scatter(
        h_2[:, 0],
        h_2[:, 1],
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
        h_2[:, 0],
        h_2[:, 1],
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