import os
import numpy as np
import tensorflow as tf
import gymnasium as gym


# 画像を64×64に変換
def preprocess(obs):

    x = tf.convert_to_tensor(obs)

    x = tf.image.resize(x, (64, 64))

    # 保存容量を抑えるためuint8にする（画像の値を8ビット整数で保存）
    x = tf.cast(tf.round(x), tf.uint8)

    return x.numpy()


# ランダムな行動を生成
def random_action(rng):

    steering = rng.uniform(-1.0, 1.0)
    gas = rng.uniform(0.3, 1.0)
    brake = rng.uniform(0.0, 0.2)

    return np.array(
        [steering, gas, brake],
        dtype=np.float32
    )


# 1エピソードのデータ収集
def collect_episode(env, rng, max_steps=1000):

    obs, info = env.reset()

    # 最初の観測画像を保存
    images = [preprocess(obs)]
    actions = []
    rewards = []

    for step in range(max_steps):

        # 行動を生成（方策はランダム）
        action = random_action(rng)

        # 環境を1ステップ進める
        next_obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        # 行動と報酬を保存
        actions.append(action)
        rewards.append(reward)

        # 次状態の画像を保存
        images.append(preprocess(next_obs))

        obs = next_obs

        if terminated or truncated:
            break

    return (
        np.stack(images),
        np.stack(actions),
        np.asarray(rewards, dtype=np.float32)
    )


def main():

    num_episodes = 10
    max_steps = 1000
    seed = 42

    save_dir = "data/carRacing"
    os.makedirs(save_dir, exist_ok=True)

    rng = np.random.default_rng(seed)

    # 画面表示無しで環境を生成
    env = gym.make(
        "CarRacing-v3",
        render_mode=None,
        continuous=True
    )

    try:
        for episode in range(num_episodes):

            images, actions, rewards = collect_episode(
                env,
                rng,
                max_steps=max_steps
            )

            # エピソードごとに保存
            save_path = os.path.join(
                save_dir,
                f"episode_{episode:04d}.npz"
            )

            np.savez_compressed(
                save_path,
                images=images,
                actions=actions,
                rewards=rewards
            )

            print(
                f"Episode {episode + 1}/{num_episodes} "
                f"Steps: {len(actions)} "
                f"Reward: {np.sum(rewards):.2f}"
            )

            print("images:", images.shape)
            print("actions:", actions.shape)

    finally:
        env.close()


if __name__ == "__main__":
    main()