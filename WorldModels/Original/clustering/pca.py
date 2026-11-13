
import glob
import numpy as np
from sklearn.decomposition import PCA

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

        print("z_shape:",z.shape)

        all_z.append(z)

    z = np.concatenate(all_z, axis=0)

    print("z_shape:",z.shape)

    pca = PCA(n_components=2)

    z_pca = pca.fit_transform(z)

    print("z_pca_shape:", z_pca.shape)
    print("寄与率:", pca.explained_variance_ratio_)


    # 全主成分を計算
    pca_all = PCA()
    pca_all.fit(z)

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