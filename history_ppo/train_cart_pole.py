#!/usr/bin/env python

##
#
# Train a policy with PPO that maps from a history of observations to a control
# action
#
##

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from policy_network import CustomActorCriticPolicy
from env_with_history import ObservationHistoryEnv

# Number of timesteps of observations to record
history_length = 3

# Wrapping env in a monitor allows us to print some useful stuff
env = Monitor(ObservationHistoryEnv(history_length), None)
vec_env = DummyVecEnv([lambda: env])

# set up the model (a.k.a. controller)
model = PPO(CustomActorCriticPolicy, vec_env, verbose=1)

# Do the learning
model.learn(total_timesteps=30000)

# Save the model
model.save("cart_pole")

