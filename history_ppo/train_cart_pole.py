#!/usr/bin/env python

##
#
# Train a policy with PPO that maps from a history of observations to a control
# action
#
##

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from policy_network import CustomActorCriticPolicy

# Create an environment that returns a history of observations
def make_env_with_observation_history(history_length):
    class ObservationHistoryEnv(gym.Env):
        def __init__(self, history_length):
            self.env = gym.make("InvertedPendulum-v4", render_mode="human")
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

    env = Monitor(ObservationHistoryEnv(history_length), None)
    return DummyVecEnv([lambda: env])

if __name__=="__main__":
    vec_env = make_env_with_observation_history(4)

    # set up the model (a.k.a. controller)
    model = PPO(CustomActorCriticPolicy, vec_env, verbose=1)

    # Do the learning
    model.learn(total_timesteps=30000)

    # Save the model
    model.save("cart_pole")

