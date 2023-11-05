"""Registers the internal gym envs then loads the env plugins for module using the entry point."""
from gymnasium.envs.registration import register

register(
    id="HEMS",
    entry_point="envs.pecan_street:HEMSEnv",
    max_episode_steps=24,
)
