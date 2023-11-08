from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

save_dir = Path("result/backup/baselines/baseline-0")
env_num = 10

data = np.vstack([np.load(save_dir / f"point-{i}" / "episode_reward_list.npy") for i in range(env_num)])
label = [i for i in range(env_num)]
plt.plot(data.mean(axis=0), label="Avg", marker="^")
# plt.plot(data.transpose(), label=label)
plt.legend(), plt.grid()
plt.show()
