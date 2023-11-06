"""Wrapper for limiting the time steps of an environment."""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import gymnasium as gym


if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec


class MaxEpisodeStepsLimit(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: int = None,
    ):
        if max_episode_steps is None:
            day_range = env.unwrapped.day_range
            max_episode_steps = (day_range[1] - day_range[0]) * 24

        gym.utils.RecordConstructorArgs.__init__(self, max_episode_steps=max_episode_steps)
        gym.Wrapper.__init__(self, env)

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    @property
    def spec(self) -> EnvSpec | None:
        """Modifies the environment spec to include the `max_episode_steps=self._max_episode_steps`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            env_spec = deepcopy(env_spec)
            env_spec.max_episode_steps = self._max_episode_steps

        self._cached_spec = env_spec
        return env_spec
