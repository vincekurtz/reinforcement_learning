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

# Set up the environment
vec_env = make_vec_env("InvertedPendulum-v4", n_envs=1)

# Load the trained model
model = PPO.load("cart_pole")

# Hackily set the initial state
vec_env.reset()
q0 = np.array([0, 0.1])
v0 = np.array([-0.1, 0.2])
vec_env.envs[0].env.env.env.env.set_state(q0, v0)
obs, _, _, _ = vec_env.step(np.array([[0]]))

# Run a little simulation
for i in range(500):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")

