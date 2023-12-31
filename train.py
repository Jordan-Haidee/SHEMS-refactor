import argparse
import copy
import random
from datetime import datetime
from pathlib import Path

import addict
import numpy as np
import pytoml
import rich
import torch
from rich.panel import Panel

from model import FedDDPG

# read config
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.toml")
parser.add_argument("--no-heter", action="store_true")
args = parser.parse_args()
with open(Path(__file__).parent / args.config, encoding="utf-8") as f:
    config = addict.Dict(pytoml.load(f))

# set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

exp_start_time = datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")
save_dir = Path(__file__).parent / f"{config.save_dir}-{exp_start_time}"
save_dir.mkdir(parents=True)

heter_set = (
    np.random.uniform(0, 1, (config.env_num, config.heter_num))
    if args.no_heter is False
    else np.full((config.env_num, config.heter_num), 0.5)
)
backup_config = {**copy.deepcopy(config), "heter_set": heter_set.tolist()}
with open(save_dir / "config_backup.toml", "w") as f:
    pytoml.dump(backup_config, f)

ddpg_config_list = [
    {
        "env": config.env,
        "id": i,
        "heter": heter_set[i],
        "seed": config.seed + i,
        "hidden_dim": config.hidden_dim,
        "lr": config.lr,
        "gamma": config.gamma,
        "tau": config.tau,
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
    merge_num=config.merge_num,
    merge_interval=config.merge_interval,
    episode_num_eval=config.episode_num_eval,
    save_dir=save_dir,
    merge_target=config.merge_target,
    device=config.device,
)
rich.print(Panel(f"Training..., result saves to: \n{save_dir.absolute()}"))
model.train()
