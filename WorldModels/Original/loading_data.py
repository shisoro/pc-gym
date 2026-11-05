import numpy as np

data = np.load("data/carRacing/episode_0000.npz")

images = data["images"]
actions = data["actions"]
rewards = data["rewards"]

print("images:", images.shape)
print("actions:", actions.shape)
print("rewards:", rewards.shape)

print("最初の行動:", actions[0])
print("累積報酬:", np.sum(rewards))
