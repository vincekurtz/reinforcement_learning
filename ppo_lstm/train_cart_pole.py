#!/usr/bin/env python

##
#
# Train a recurrent (LSTM) policy to control a simulated cart-pole
#
##

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.env_util import make_vec_env

from sb3_contrib import RecurrentPPO
from custom_policy import CustomRecurrentActorCriticPolicy


vec_env = make_vec_env("InvertedPendulum-v4", n_envs=1)

model = RecurrentPPO(CustomRecurrentActorCriticPolicy, vec_env, verbose=1)

print(model.policy)

model.learn(total_timesteps=10)

#model.learn(total_timesteps=30000)
#model.save("cart_pole_lstm")

