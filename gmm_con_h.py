
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


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

    print("h_shape:",h.shape)

    #混合ガウスモデルによるクラスタリング


    #想定するクラスター数のリスト
    components_num = [2, 3, 4, 5, 6]

    #クラスター数を1,2,...と振ってGMMを行い、BICを算出
    for i in components_num:
        gmm = GaussianMixture(n_components=i,covariance_type='full') #混合ガウスモデルを定義してフィット
        gmm.fit(h)
        BIC = gmm.bic(h) #BICの計算（GaussianMixtureのBICメソッドを用いる）

        #クラスターを可視化（2次元で表示）
        plt.scatter(h[:, 0], h[:, 1], c=gmm.predict(h), s=50)
        plt.xlabel("PC1")
        plt.ylabel("PC2")

        #クラスタの中心をプロット
        centers = gmm.means_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
        plt.title('The number of cluster : '+str(i)+'  (BIC:'+str(round(BIC,1))+')')

        #レイアウトを調整
        plt.tight_layout()
        
        plt.show()


if __name__ == "__main__":
    main()  