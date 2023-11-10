"""Registers the internal gym envs then loads the env plugins for module using the entry point."""
from gymnasium.envs.registration import register, WrapperSpec

register(
    id="HEMS-heter",
    entry_point="envs.envs:HEMSEnv",
    additional_wrappers=(
        WrapperSpec(
            name="MaxEpisodeStepsLimit",
            entry_point="envs.wrappers:MaxEpisodeStepsLimit",
            kwargs={"max_episode_steps": None},
        ),
        WrapperSpec(
            name="ActionRestrict",
            entry_point="envs.wrappers:ActionRestrict",
            kwargs={"args": None},
        ),
    ),
)
