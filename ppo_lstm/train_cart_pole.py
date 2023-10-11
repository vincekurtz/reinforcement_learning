#!/usr/bin/env python

##
#
# Train a recurrent (LSTM) policy to control a simulated cart-pole
#
##

import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("InvertedPendulum-v4", n_envs=1)

model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=1)

model.learn(total_timesteps=30000)

vec_env = model.get_env()

model.save("cart_pole_lstm")

