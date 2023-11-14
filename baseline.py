import argparse
import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import pytoml
from addict import Dict

from model import FedDDPG

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.toml")
args = parser.parse_args()

with open(Path(__file__).parent / args.config, encoding="utf-8") as f:
    config = Dict(pytoml.load(f))

exp_start_time = datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")
np.random.seed(config.seed)
save_dir = Path(config.save_dir).parent / f"baseline-{exp_start_time}"
save_dir.mkdir(parents=True)

heter_set = np.random.uniform(0, 1, (config.env_num, config.heter_num))
backup_config = {**copy.deepcopy(config), "heter_set": heter_set.tolist()}
with open(save_dir / "config_backup.toml", "w") as f:
    pytoml.dump(backup_config, f)

ddpg_config_list = [
    {
        "env": config.env,
        "id": i,
        "heter": heter_set[i],
        "seed": config.seed + i,
        "embedding_init": None,
        "lr": config.lr,
        "gamma": config.gamma,
        "hidden_dim": config.hidden_dim,
        "buffer_capicity": config.buffer_capicity,
        "buffer_init_ratio": config.buffer_init_ratio,
        "batch_size": config.batch_size,
        "train_batchs": config.merge_num * config.merge_interval,
        "save_dir": save_dir / f"point-{i}",
        "device": config.device,
    }
    for i in range(config.env_num)
]

model = FedDDPG(
    point_configs=ddpg_config_list,
    merge_interval=config.merge_interval,
    merge_num=config.merge_num,
    episode_num_eval=config.episode_num_eval,
    save_dir=save_dir,
    device=config.device,
    merge_target=config.merge_target,
)
print(f"Training baseline..., result saves to: \n{save_dir.absolute()}")
model.train_baseline()
