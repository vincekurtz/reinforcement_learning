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
from gymnasium.wrappers import TimeLimit, OrderEnforcing, PassiveEnvChecker

SEED = 0
set_random_seed(SEED, using_cuda=True)

def make_environment():
    env = SlowManifoldEnv(mu=-0.5, lam=-0.1, dt=5e-2)
    env.action_space.seed(SEED)

    env = Monitor(TimeLimit(OrderEnforcing(PassiveEnvChecker(env)), max_episode_steps=200))
    vec_env = DummyVecEnv([lambda: env])
    vec_env.seed(SEED)
    return vec_env

def simulate(policy_fcn = lambda obs: np.array([[0.0]]), num_traj = 1):
    """
    Given a function that maps observations to actions, simulate the environment
    under that policy and plot the resulting trajectory.
    """
    env = make_environment()

    for _ in range(num_traj):
        obs = env.reset()
        states = [obs]
        for t in range(199):
            action = policy_fcn(obs)
            obs, _, done, _ = env.step(action)
            states.append(obs)
        states = np.array(states).reshape(-1, 2)
        plt.plot(states[:, 0], states[:, 1], 'o-')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

def train():
    """
    Train the model with PPO and save it to disk.
    """
    vec_env = make_environment()

    model = PPO("MlpPolicy", vec_env, gamma=0.98, learning_rate=1e-3,
                tensorboard_log="/tmp/slow_manifold_tensorboard/",
                policy_kwargs=dict(net_arch=[64, 64]),
                verbose=1)
    print(model.policy)

    model.learn(total_timesteps=100_000)

    # Save the model to disk
    model.save("trained_models/slow_manifold")

def test():
    """
    Load a trained model from disk and simulate it.
    """
    model = PPO.load("trained_models/slow_manifold")

    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    #simulate(num_traj=10)
    simulate(policy, num_traj=10)


if __name__=="__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        test()