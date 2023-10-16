import gymnasium as gym
import numpy as np

class ObservationHistoryEnv(gym.Env):
    """
    A gym environment for the cart-pole that returns a history of
    observations.
    """
    def __init__(self, history_length, render_mode=None):
        self.env = gym.make("InvertedPendulum-v4", render_mode=render_mode)
        self.history_length = history_length
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.history_length * self.env.observation_space.shape[0],), dtype=np.float32
        )

        self.action_space = self.env.action_space
        self.history_buffer = None

    def reset(self, **kwargs):
        # Reset the environment and observation history buffer
        obs, info = self.env.reset(**kwargs)
        self.history_buffer = [obs] * self.history_length
        return self._get_observation(), info

    def step(self, action):
        # Take a step in the environment and update the observation history buffer
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history_buffer.pop(0)
        self.history_buffer.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        return np.concatenate(self.history_buffer)

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

