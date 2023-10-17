#!/usr/bin/env python

##
#
# Test a trained RL policy, loaded from disk.
#
##

import gymnasium as gym
import torch
from train import PolicyNetwork

# Set up the environment
env = gym.make('Pendulum-v1', render_mode="human")

# Load the policy network from disk
policy_network = PolicyNetwork(3,1)
policy_network.load_state_dict(torch.load('policy.pt'))

# Run the policy
obs, _ = env.reset()
for t in range(500):
    env.render()
    action, _ = policy_network(torch.Tensor(obs))
    obs, _, terminated, truncated, _ = env.step(action.detach().numpy())
    if terminated or truncated:
        break
