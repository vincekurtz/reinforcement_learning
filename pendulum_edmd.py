#!/usr/bin/env python

##
#
# Try to analyze the learned lifted dynamics of the pendulum, to see whether
# we've learned a Koopman representation for the closed-loop system.
#
##

import gymnasium as gym
from stable_baselines3 import PPO

import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the learned model
model = PPO.load("trained_models/pendulum")
phi = model.policy.mlp_extractor.lifting_function
print(phi)
