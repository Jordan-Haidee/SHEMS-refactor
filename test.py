from argparse import ArgumentParser
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytoml
import torch
from addict import Dict

from model import Actor

parser = ArgumentParser(description="Test FedDDPG Algo & Baseline")
parser.add_argument("save_dir", type=str, help="Exp result save directory")
parser.add_argument("--valid", action="store_true", help="Whether is validation use training data or not")
args = parser.parse_args()

save_dir = Path(args.save_dir)
with open(save_dir / "config_backup.toml") as f:
    config = Dict(pytoml.load(f))
env = gym.make(config.env)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high
total_r = []
for i in range(config.env_num):
    env = gym.make(config.env, heter=config.heter_set[i], is_test=True if not args.valid else False)
    actor = Actor(state_dim, 400, action_bound)
    try:
        actor.load_state_dict(torch.load(save_dir / f"point-{i}" / "latest.pt").get("actor"))
    except TypeError:
        actor.load_state_dict(torch.load(save_dir / f"point-{i}" / "latest.pt").get("weights").get("actor"))
    s, _ = env.reset()
    point_r = 0
    with torch.no_grad():
        while True:
            a = actor(torch.from_numpy(env.unwrapped.normalize_state(s))).numpy()
            ns, r, t1, t2, info = env.step(a)
            print(a.round(4), ns.round(4), round(info["c1"], 4), round(info["c2"], 4), round(info["c3"], 4))
            s = ns
            point_r += r
            if t1 or t2:
                total_r.append(point_r)
                break
total_r = np.array(total_r)
print(total_r.round(4))
print(total_r.mean())
