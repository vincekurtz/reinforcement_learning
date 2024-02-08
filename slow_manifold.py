#!/usr/bin/env python

##
#
# Train or test a policy on the slow manifold system.
#
##

import sys
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

from policies import KoopmanPolicy
from slow_manifold_env import SlowManifoldEnv

SEED = 0
set_random_seed(SEED, using_cuda=True)

def make_environment():
    env = SlowManifoldEnv(mu=-0.1, lam=-1.0, dt=5e-2)
    env.action_space.seed(SEED)

    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env.seed(SEED)
    return vec_env

def simulate(policy_fcn = lambda obs: np.array([0.0]), num_traj = 1):
    """
    Given a function that maps observations to actions, simulate the environment
    under that policy and plot the resulting trajectory.
    """
    env = make_environment()

    for _ in range(num_traj):
        obs = env.reset()
        states = [obs]
        for t in range(200):
            action = policy_fcn(obs)[np.newaxis]  # for batch dimension
            obs, _, done, _ = env.step(action)
            states.append(obs)
        states = np.array(states).reshape(-1, 2)
        plt.plot(states[:, 0], states[:, 1], 'o-')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()


if __name__=="__main__":
    simulate(num_traj=10)