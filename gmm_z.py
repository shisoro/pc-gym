
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


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

    #混合ガウスモデルによるクラスタリング


    #想定するクラスター数のリスト
    components_num = [1, 3, 5, 7,]

    #クラスター数を1,2,...と振ってGMMを行い、BICを算出
    for i in components_num:
        gmm = GaussianMixture(n_components=i,covariance_type='full') #混合ガウスモデルを定義してフィット
        gmm.fit(z)
        BIC = gmm.bic(z) #BICの計算（GaussianMixtureのBICメソッドを用いる）

        #クラスターを可視化（2次元で表示）
        plt.scatter(z[:, 0], z[:, 1], c=gmm.predict(z), s=50)
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