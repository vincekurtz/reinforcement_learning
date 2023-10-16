#!/usr/bin/env python

##
#
# Load a trained policy and use it to control a simulated cart-pole
#
##

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env_with_history import ObservationHistoryEnv

# Create the environment
env = ObservationHistoryEnv(render_mode="human")
vec_env = DummyVecEnv([lambda: env])

# Load the trained model
model = PPO.load("cart_pole")

# Run a little simulation
obs = vec_env.reset()
for i in range(500):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()

