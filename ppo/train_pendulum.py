#!/usr/bin/env python

##
#
# Train a policy with standard PPO on a simple inverted pendulum example
#
##

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from policy_network import CustomActorCriticPolicy
from envs import PendulumWithObservationHistory

# Set up the environment (a.k.a. plant)
#vec_env = make_vec_env(PendulumWithObservationHistory, n_envs=1,
#                       env_kwargs={"history_length": 1})
vec_env = make_vec_env("Pendulum-v1", n_envs=1)

# set up the model (a.k.a. controller)
model = PPO(CustomActorCriticPolicy, vec_env, gamma=0.98, learning_rate=1e-3, 
            tensorboard_log="/tmp/pendulum_tensorboard/",
            verbose=1)

# Do the learning
model.learn(total_timesteps=200_000)

# Save the model
model.save("pendulum")

