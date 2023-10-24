##
#
# Some custom gym environments
#
##

import gymnasium as gym
import numpy as np

class PendulumWithObservationHistory(gym.Env):
    """
    Identical to the 'Pendulum-v1' environment, but observations are
        [cos(theta), sin(theta), theta_dot] 
    from the previous N steps.
    """
    def __init__(self, history_length=10, render_mode=None):
        self.history_length = history_length
        self.env = gym.make("Pendulum-v1", render_mode=render_mode)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3*self.history_length,), dtype=np.float32
        )
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        self.history = np.zeros((self.history_length, 3))
        obs, info = self.env.reset(**kwargs)
        self.history[0] = obs
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history = np.roll(self.history, shift=1, axis=0)
        self.history[0] = obs
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return self.history.flatten()