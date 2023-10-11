#!/usr/bin/env python

##
#
# Load a trained LSTM policy and use it to control a simulated cart-pole
#
##

import gymnasium as gym
import numpy as np
import time

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env

num_envs = 1
vec_env = make_vec_env("InvertedPendulum-v4", n_envs=num_envs)

model = RecurrentPPO.load("cart_pole_lstm")

obs = vec_env.reset()
# cell and hidden state of the LSTM
lstm_states = None
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    episode_starts = dones
    vec_env.render("human")

