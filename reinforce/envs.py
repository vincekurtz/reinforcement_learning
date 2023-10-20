##
#
# Some custom gym environments
#
##

import gymnasium as gym
import numpy as np

class InvertedPendulumNoVelocity(gym.Env):
    """
    The 'InvertedPendulum-v4' environment, but with velocity observations
    masked. This gives us an example where we need to use output feedback.
    """
    def __init__(self, render_mode=None):
        self.env = gym.make('InvertedPendulum-v4', render_mode=render_mode)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = self.env.action_space

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs[0:2], info
    
    def render(self):
        self.env.render()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs[0:2], reward, terminated, truncated, info
    
class PendulumFixedReset(gym.Env):
    """
    The 'Pendulum-v1' environment, but always starts with the pendulum facing
    down at zero velocity.
    """
    def __init__(self, render_mode=None):
        self.env = gym.make("Pendulum-v1", render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        # Fix the initial state
        theta, thetadot = np.pi, 0.0
        self.env.unwrapped.state = np.array([theta, thetadot])
        obs = np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        return obs, {}
    
    def render(self):
        self.env.render()

    def step(self, action):
        return self.env.step(action)
