import numpy as np

import gymnasium as gym
from gymnasium import spaces

class SlowManifoldEnv(gym.Env):
    """
    A gymnasium environment for the slow manifold system often used as a Koopman
    demo:

        ẋ₁ = μ x₁ 
        ẋ₂ = λ(x₂ − x₁²) + u

    The goal is to stabilize the system at the origin (0, 0).
    """
    def __init__(self, mu=1.0, lam=1.0, dt=1e-2):
        # System parameters
        self.mu = mu
        self.lam = lam
        self.dt = dt

        x_max = 1.0
        u_max = 0.1
        self.action_space = spaces.Box(
            low=-u_max, high=u_max, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-x_max, high=x_max, shape=(2,), dtype=np.float32)
    
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
        self.state = self.observation_space.sample()
        return self._get_obs(), {}

    def step(self, u):
        x1, x2 = self.state 
        u = np.clip(u, -1.0, 1.0)[0]

        # Advance the system state
        x1_dot = self.mu * x1
        x2_dot = self.lam * (x2 - x1**2) + u
        self.state = np.array([x1 + x1_dot * self.dt, 
                               x2 + x2_dot * self.dt])

        # Compute the cost to drive the system to the origin
        cost = x1**2 + x2**2 + u**2

        return self._get_obs(), -cost, False, False, {}

    def _get_obs(self):
        return self.state

    def render(self):
        pass

    def close(self):
        pass