import numpy as np

import gymnasium as gym


class ActionRestrict(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, args=None):
        gym.utils.RecordConstructorArgs.__init__(self, args=args)
        gym.ActionWrapper.__init__(self, env)

    def action(self, a):
        p_ess, p_hvac = a
        p_solar, p_load, ess_level, temp_outdoor, temp_indoor, price, _ = self.env.unwrapped.state
        if p_ess >= 0:
            p_ess = np.clip(
                p_ess,
                0,
                min(
                    (self.env.unwrapped.ess_level_max - ess_level) / self.env.unwrapped.eta_ess,
                    self.env.unwrapped.p_ess_max,
                ),
            )
        else:
            p_ess = -np.clip(
                -p_ess,
                0,
                min(
                    (ess_level - self.env.unwrapped.ess_level_min) * self.env.unwrapped.eta_ess,
                    self.env.unwrapped.p_ess_max,
                ),
            )
        if temp_indoor <= self.env.unwrapped.T_min:
            p_hvac = 0
        if temp_indoor > self.env.unwrapped.T_max:
            p_hvac = np.clip(p_hvac, 0.1, self.env.unwrapped.p_hvac_max)
        return np.array([p_ess, p_hvac])
