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

# Wrapping env in a monitor allows us to print some useful stuff
env = Monitor(ObservationHistoryEnv(), None)
vec_env = DummyVecEnv([lambda: env])

# set up the model (a.k.a. controller)
model = PPO(CustomActorCriticPolicy, vec_env, verbose=1)

print(f"Training a policy with {len(model.policy.parameters_to_vector())} parameters")

# Do the learning
model.learn(total_timesteps=150000)

# Save the model
model.save("cart_pole")

