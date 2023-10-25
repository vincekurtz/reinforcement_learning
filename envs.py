##
#
# Some custom gym environments
#
##

import gymnasium as gym
import numpy as np

class EnvWithObservationHistory(gym.Env):
    """
    A simple gym environment wrapper that outputs a history of observations
    instead of just the current one.
    """
    def __init__(self, env_name, history_length, render_mode=None):
        self.history_length = history_length
        self.env = gym.make(env_name, render_mode=render_mode)

        # The environment must have vector observations
        assert isinstance(self.env.observation_space, gym.spaces.Box)
        assert len(self.env.observation_space.shape) == 1
        
        self.observation_size = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.observation_size*self.history_length,), 
            dtype=np.float32
        )
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        self.history = np.zeros((self.history_length, self.observation_size))
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
