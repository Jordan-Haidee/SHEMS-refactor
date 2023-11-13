from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
import pandas as pd
import pytoml
import torch
from addict import Dict
from matplotlib import pyplot as plt
from rich import print as rprint
from rich.table import Table
from tqdm import trange

from model import Actor

Record = namedtuple("Record", ["s", "a", "r", "d", "c1", "c2", "c3"])
parser = ArgumentParser(description="Test FedDDPG Algo & Baseline")
parser.add_argument("save_dir", type=str, help="Exp result save directory")
parser.add_argument("--valid", action="store_true", help="Whether is validation use training data or not")
parser.add_argument(
    "--report-dir",
    default=f"result/report-{datetime.now().strftime(r'%Y-%m-%d-%H-%M-%S')}",
    help="The directory of generated report saved",
)
args = parser.parse_args()
report_dir = Path(args.report_dir)
report_dir.mkdir(parents=True)
save_dir = Path(args.save_dir)
with open(save_dir / "config_backup.toml") as f:
    config = Dict(pytoml.load(f))
env = gym.make(config.env)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_scope = list(zip(env.action_space.low, env.action_space.high))

point_record_list: List[List[Record]] = []
# collect info
for i in trange(config.env_num, ncols=80, desc="collecting info->"):
    env = gym.make(config.env, heter=config.heter_set[i], is_test=True if not args.valid else False)
    actor = Actor(state_dim, config.hidden_dim, action_dim, action_scope)
    try:
        actor.load_state_dict(torch.load(save_dir / f"point-{i}" / "latest.pt").get("actor"))
    except TypeError:
        actor.load_state_dict(torch.load(save_dir / f"point-{i}" / "latest.pt").get("weights").get("actor"))
    traj: List[Record] = []
    s, _ = env.reset()
    with torch.no_grad():
        while True:
            a = actor(torch.from_numpy(env.unwrapped.normalize_state(s))).numpy()
            ns, r, t1, t2, info = env.step(a)
            c1, c2, c3 = info.values()
            traj.append(Record(s, env.unwrapped.clip_action(a), r, t1 or t2, c1, c2, c3))
            s = ns
            if t1 or t2:
                point_record_list.append(traj)
                break

# save info
table_list: List[pd.DataFrame] = []
for i in trange(config.env_num, ncols=80, desc="summarizing info->"):
    traj = point_record_list[i]
    detailed_table = pd.DataFrame(
        {
            "ESS Level": [record.s[2] for record in traj],
            "Outdoor Temperature": [record.s[3] for record in traj],
            "Indoor Temperature": [record.s[4] for record in traj],
            "Price": [record.s[5] for record in traj],
            "Time Slot": [record.s[6] for record in traj],
            "ESS Power": [record.a[0] for record in traj],
            "HVAC Power": [record.a[1] for record in traj],
            "Total Cost": [-record.r for record in traj],
            "Energy Cost": np.array([record.c1 for record in traj]) + np.array([record.c2 for record in traj]),
            "Discomfort Cost": [record.c3 for record in traj],
            "C1": [record.c1 for record in traj],
            "C2": [record.c2 for record in traj],
            "C3": [record.c3 for record in traj],
        }
    )
    Path(report_dir / f"point-{i}").mkdir()
    detailed_table.to_csv(report_dir / f"point-{i}" / "details.csv")
    summary_table = pd.DataFrame(
        {
            "Total Cost": [detailed_table["Total Cost"].sum()],
            "Energy Cost": [detailed_table["Energy Cost"].sum()],
            "Discomfort Cost": [detailed_table["Discomfort Cost"].sum()],
            "Maingrid Cost": [detailed_table["C1"].sum()],
            "ESS Cost": [detailed_table["C2"].sum()],
        },
    )
    summary_table.to_csv(report_dir / f"point-{i}" / "summary.csv")
    table_list.append(detailed_table)
    fig, axes = plt.subplots(6, 1, figsize=(90, 20))
    xticks = np.arange(0, 24 * (env.unwrapped.day_duration + 1), 24)
    xlim = [xticks.min(), xticks.max()]
    for k, title in enumerate(
        [
            "Price",
            "ESS Level",
            "ESS Power",
            "Outdoor Temperature",
            "Indoor Temperature",
            "HVAC Power",
        ]
    ):
        detailed_table[title].plot(
            ax=axes[k],
            title=title,
            grid=True,
            xticks=xticks,
            xlim=xlim,
        )
    fig.savefig(report_dir / f"point-{i}" / "run_details.svg")

# analyze avg performance
avg_total_cost = np.array([table["Total Cost"].sum() for table in table_list]).mean()
avg_energy_cost = np.array([table["Energy Cost"].sum() for table in table_list]).mean()
avg_discomfort_cost = np.array([table["Discomfort Cost"].sum() for table in table_list]).mean()
avg_maingrid_cost = np.array([table["C1"].sum() for table in table_list]).mean()
avg_ess_cost = np.array([table["C2"].sum() for table in table_list]).mean()

# show brief info
table = Table(title=f"Average Performance of {config.env_num} Environments")
table.add_column("Total")
table.add_column("Energy Cost")
table.add_column("Comfort Cost")
table.add_column("Maingrid Cost")
table.add_column("ESS Cost")
table.add_row(
    f"{round(avg_total_cost,4)}",
    f"{round(avg_energy_cost,4)}",
    f"{round(avg_discomfort_cost,4)}",
    f"{round(avg_maingrid_cost,4)}",
    f"{round(avg_ess_cost.mean(),4)}",
)
rprint(table)
