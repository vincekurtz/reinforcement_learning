#!/usr/bin/env python

##
#
# Train a policy with standard PPO on a simple inverted pendulum example
#
##

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from policy_network import CustomActorCriticPolicy
from envs import PendulumWithObservationHistory

# Try to make things deterministic
set_random_seed(1, using_cuda=True)

# Set up the environment (a.k.a. plant)
vec_env = make_vec_env(PendulumWithObservationHistory, n_envs=1,
                       env_kwargs={"history_length": 10})

# set up the model (a.k.a. controller)
model = PPO(CustomActorCriticPolicy, vec_env, gamma=0.98, learning_rate=1e-3, 
            tensorboard_log="/tmp/pendulum_tensorboard/",
            verbose=1)

# Print how many parameters this thing has
num_params = sum(p.numel() for p in model.policy.parameters())
print("Training a policy with {num_params} parameters")

# Do the learning
model.learn(total_timesteps=200_000)

# Save the model
model.save("pendulum")

