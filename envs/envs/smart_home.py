import functools
from pathlib import Path
from typing import Optional, Union

import addict
import gymnasium as gym
import numpy as np
import pandas as pd
import pytoml
import torch


class HEMSEnv(gym.Env):
    def __init__(
        self,
        config_path: str = Path(__file__).parent / "default.toml",
        heter: np.ndarray = np.array([0.5, 0.5, 0.5]),
        seed: int = None,
        is_test: bool = False,
    ) -> None:
        with open(config_path, encoding="utf-8") as f:
            self.config = addict.Dict(pytoml.load(f))
        self.eta_ess = self.config.eta_ess
        self.ess_level_init = self.config.ess_level_init
        self.ess_level_max = (self.config.ess_level_max - 1.8) + 3.6 * heter[0]
        self.ess_level_min = (self.config.ess_level_min - 0.18) + 0.36 * heter[1]
        self.ess_aging_cost = self.config.ess_aging_cost
        self.p_ess_max = self.config.p_ess_max
        self.p_hvac_max = self.config.p_hvac_max
        self.T_min = self.config.T_min
        self.T_max = self.config.T_max
        self.beta_hvac = self.config.beta_hvac
        self.epsilon_hvac = self.config.epsilon_hvac
        self.eta_hvac = self.config.eta_hvac
        self.A_hvac = (self.config.A_hvac - 0.02) + 0.04 * heter[2]
        self.temp_indoor_init = self.config.temp_indoor_init
        self.solar_data_path = self.config.solar_data_path
        self.load_data_path = self.config.load_data_path
        self.temp_outdoor_data_path = self.config.temp_outdoor_data_path
        self.price_data_path = self.config.price_data_path
        # ------------------------------------------------------------------
        self.day_range = self.config.day_range
        if is_test is True:
            self.day_range = [self.day_range[-1], self.day_range[-1] + 31]
        self.day_duration = self.day_range[1] - self.day_range[0]
        self.price_ratio = self.config.price_ratio

        # 加载数据
        self.solar_data_table = pd.read_csv(Path(__file__).parent / self.solar_data_path)
        self.load_data_table = pd.read_csv(Path(__file__).parent / self.load_data_path)
        self.temp_outdoor_data_table = pd.read_csv(Path(__file__).parent / self.temp_outdoor_data_path)
        self.price_data_table = pd.read_csv(Path(__file__).parent / self.price_data_path)

        # 设置动作空间和观测空间
        self.action_space = gym.spaces.Box(
            low=np.array([-self.p_ess_max, 0]),
            high=np.array([self.p_ess_max, self.p_hvac_max]),
        )
        if seed is not None:
            self.action_space.seed(seed)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, -np.finfo(np.float32).max, 0, 0]),
            high=np.array(
                [
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                ]
            ),
            dtype=np.float32,
        )

    @property
    @functools.lru_cache
    def data_scope(self):
        return np.array(
            [
                [self.solar_data_table.min().min(), self.solar_data_table.max().max()],
                [self.load_data_table.min().min(), self.load_data_table.max().max()],
                [self.ess_level_min, self.ess_level_max],
                [
                    self.temp_outdoor_data_table.min().min(),
                    self.temp_outdoor_data_table.max().max(),
                ],
                [self.T_min, self.T_max],
                [self.price_data_table.min().min(), self.price_data_table.max().max()],
                [0, 23],
            ]
        )

    def normalize_state(self, s: Union[np.ndarray, torch.Tensor]):
        assert len(s.shape) == 1
        s_ = np.zeros_like(s) if isinstance(s, np.ndarray) else torch.zeros_like(s)
        for i in range(self.observation_space.shape[0]):
            s_[i] = (s[i] - self.data_scope[i][0]) / (self.data_scope[i][1] - self.data_scope[i][0])
        return s_

    def clip_action(self, a):
        p_ess, p_hvac = a
        p_solar, p_load, ess_level, temp_outdoor, temp_indoor, price, _ = self.state
        if p_ess >= 0:
            p_ess = np.clip(
                p_ess,
                0,
                min(
                    (self.ess_level_max - ess_level) / self.eta_ess,
                    self.p_ess_max,
                ),
            )
        else:
            p_ess = -np.clip(
                -p_ess,
                0,
                min(
                    (ess_level - self.ess_level_min) * self.eta_ess,
                    self.p_ess_max,
                ),
            )
        if temp_indoor <= self.T_min:
            p_hvac = 0
        if temp_indoor > self.T_max:
            p_hvac = np.clip(p_hvac, 0.1, self.p_hvac_max)
        return np.array([p_ess, p_hvac])

    def step(self, a: np.ndarray):
        p_solar, p_load, ess_level, temp_outdoor, temp_indoor, price, _ = self.state
        p_ess, p_hvac = self.clip_action(a)
        # 计算ESS下一时隙的储能
        ess_level_next = ess_level + (p_ess * self.eta_ess if p_ess > 0 else p_ess / self.eta_ess)
        ess_level_next = np.clip(ess_level_next, self.ess_level_min, self.ess_level_max)
        # 计算下一时隙的室内温度
        temp_indoor_next = self.epsilon_hvac * temp_indoor + (1 - self.epsilon_hvac) * (
            temp_outdoor - self.eta_hvac * p_hvac / self.A_hvac
        )
        # 计算向主电网的购电/售电量
        p_grid = p_load - p_solar + p_ess + p_hvac
        # 计算reward
        c1 = p_grid * price
        if p_grid < 0:
            c1 *= self.price_ratio
        c2 = abs(p_ess) * self.ess_aging_cost
        c3 = max(0, self.T_min - temp_indoor_next) + max(0, temp_indoor_next - self.T_max)
        r = -self.beta_hvac * (c1 + c2) - c3
        # 返回
        self.hour += 1
        if self.hour > 23:
            self.hour = 0
            self.day += 1
        hour_idx = f"{int(self.hour)}:00"
        self.state = np.array(
            [
                self.solar_data_table.iloc[self.day][hour_idx],
                self.load_data_table.iloc[self.day][hour_idx],
                ess_level_next,
                self.temp_outdoor_data_table.iloc[self.day][hour_idx],
                temp_indoor_next,
                self.price_data_table.iloc[self.day][hour_idx],
                self.hour,
            ],
            dtype=np.float32,
        )
        t1 = False
        t2 = False
        info = {"c1": c1, "c2": c2, "c3": c3}

        return self.state, r, t1, t2, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.day = self.day_range[0]
        self.state = np.array(
            [
                self.solar_data_table.iloc[self.day]["0:00"],
                self.load_data_table.iloc[self.day]["0:00"],
                self.ess_level_init,
                self.temp_outdoor_data_table.iloc[self.day]["0:00"],
                self.temp_indoor_init,
                self.price_data_table.iloc[self.day]["0:00"],
                0,
            ],
            dtype=np.float32,
        )
        self.hour = 0
        info = {"day": self.day, "hour": self.hour}
        return self.state, info


if __name__ == "__main__":
    env = HEMSEnv(Path(__file__).parent / "default.toml")
    s, _ = env.reset()
    idx = 1
    while True:
        a = env.action_space.sample()
        print(idx, s.round(4))
        s, r, t1, t2, _ = env.step(a)
        if t2:
            break
        idx += 1
