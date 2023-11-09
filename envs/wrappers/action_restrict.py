import numpy as np
import gymnasium as gym

class ActionRestrict(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, a):
        p_ess, p_hvac = a
        p_solar, p_load, ess_level, temp_outdoor, temp_indoor, price, _ = self.unwrapped.state
        if p_ess >= 0:
            p_ess = np.clip(
                p_ess,
                0,
                min(
                    (self.unwrapped.ess_level_max - ess_level) / self.unwrapped.eta_ess,
                    self.unwrapped.p_ess_max,
                ),
            )
        else:
            p_ess = -np.clip(
                -p_ess,
                0,
                min(
                    (ess_level - self.unwrapped.ess_level_min) * self.unwrapped.eta_ess,
                    self.unwrapped.p_ess_max,
                ),
            )
        if temp_indoor <= self.unwrapped.T_min:
            p_hvac = 0
        if temp_indoor > self.unwrapped.T_max:
            p_hvac = np.clip(p_hvac, 0.1, self.unwrapped.p_hvac_max)
        return np.array([p_ess, p_hvac])
