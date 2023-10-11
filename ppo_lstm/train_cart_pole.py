#!/usr/bin/env python

##
#
# Train a recurrent (LSTM) policy to control a simulated cart-pole
#
##

import numpy as np

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("InvertedPendulum-v4", n_envs=1)

model = RecurrentPPO(RecurrentActorCriticPolicy, vec_env, verbose=1,
        policy_kwargs={"shared_lstm":False, "enable_critic_lstm":False})

model.learn(total_timesteps=30000)

vec_env = model.get_env()

model.save("cart_pole_lstm")

