
import glob
import numpy as np
from sklearn.decomposition import PCA

def main():

    # 潜在変数hのファイル一覧
    files = sorted(
        glob.glob("../data/hidden/episode_*.npz")
    )

    if not files:
        raise FileNotFoundError(
            "隠れ状態hのデータが見つかりません"
        )

    all_h = []

    for episode_id, file in enumerate(files):

        with np.load(file) as data:
            h = data["h"]   

        print("h_shape:",h.shape)

        all_h.append(h)

    h = np.concatenate(all_h, axis=0)

    print("h_shape:",h.shape)

    pca = PCA(n_components=2)

    h_pca = pca.fit_transform(h)

    print("h_pca_shape:", h_pca.shape)
    print("寄与率:", pca.explained_variance_ratio_)


    # 全主成分を計算
    pca_all = PCA()
    pca_all.fit(h)

    # 各主成分の寄与率
    ratio = pca_all.explained_variance_ratio_

    # 累積寄与率
    cumulative_ratio = np.cumsum(ratio)

    for i, (r, c) in enumerate(zip(ratio, cumulative_ratio)):
        print(
            f"PC{i+1:2d}: "
            f"寄与率={r:.4f}, "
            f"累積寄与率={c:.4f}"
        )

    # 累積寄与率が一定値に達する次元数
    for threshold in [0.8, 0.9, 0.95]:
        n = np.searchsorted(cumulative_ratio, threshold) + 1
        print(f"累積寄与率{threshold:.0%}に必要な次元数: {n}")

        
if __name__ == "__main__":
    main()    