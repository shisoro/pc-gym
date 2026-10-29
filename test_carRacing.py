import gymnasium as gym
import numpy as np

# CarRacing環境を生成
env = gym.make(
    "CarRacing-v3",
    render_mode="human",
    continuous=True
)

# 環境を初期化
obs, info = env.reset(seed=42)

print("観測データの形状：", obs.shape)
print("行動空間：", env.action_space)

# 300ステップ実行
for step in range(300):

    # ステアリング・アクセル・ブレーキ
    action = np.array(
        [0.0, 0.5, 0.0],
        dtype=np.float32
    )

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    if terminated or truncated:
        obs, info = env.reset()

env.close()