#!/usr/bin/env python

##
#
# Load a trained policy and use it to control a simulated cart-pole
#
##

import gymnasium as gym
import numpy as np
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from train_cart_pole import make_env_with_observation_history

# Set up the environment
vec_env = make_env_with_observation_history(4)
#vec_env = make_vec_env("InvertedPendulum-v4", n_envs=1)

# Load the trained model
model = PPO.load("cart_pole")

# Run a little simulation
obs = vec_env.reset()
for i in range(500):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()

