#!/usr/bin/env python

##
#
# Load a trained policy and use it to control a simulated pendulum
#
##

import gymnasium as gym
import numpy as np
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs import PendulumWithObservationHistory

# Set up the environment
vec_env = make_vec_env(PendulumWithObservationHistory, n_envs=1, 
                       env_kwargs={"render_mode": "human", 
                                   "history_length": 1})

# Load the trained model
model = PPO.load("pendulum")

# Hackily set the initial state
obs = vec_env.reset()

# Run a little simulation
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")

    time.sleep(0.01)

