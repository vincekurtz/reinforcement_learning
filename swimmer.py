#!/usr/bin/env python

##
#
# Train or test a policy with standard PPO on a little swimmer robot.
#
##

import sys
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from policies import LinearPolicy
import gymnasium as gym

# Try to make things deterministic
SEED = 1
set_random_seed(SEED, using_cuda=True)

def make_environment(render_mode=None):
    """
    Set up the gym environment (a.k.a. plant). Used for both training and
    testing.
    """
    env = gym.make("Swimmer-v4", render_mode=render_mode)
    env.action_space.seed(SEED)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env.seed(SEED)

    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    return vec_env

def train(linear_system_type="parallel", num_blocks=1, timesteps=10_000):
    """
    Train the model with PPO and save it to disk.
    """
    vec_env = make_environment() 
    
    # set up the model (a.k.a. controller)
    model = PPO(LinearPolicy, vec_env, verbose=1, gamma=0.9999,
                tensorboard_log="/tmp/swimmer_tensorboard/",
                policy_kwargs={"linear_system_type": linear_system_type,
                               "num_blocks": num_blocks})

    # Print how many parameters this thing has
    num_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Training a policy with {num_params} parameters")

    # Do the learning
    model.learn(total_timesteps=timesteps)

    # Save the model
    model.save("trained_models/swimmer")

def test():
    """
    Load the trained model from disk and run a little simulation
    """
    vec_env = make_environment(render_mode="human", test_mode=True)
    model = PPO.load("trained_models/swimmer")

    obs = vec_env.reset()
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = vec_env.step(action)
        vec_env.render("human")

def evaluate(num_samples=10):
    """
    Evaluate the trained model by running it for a while and reporting the
    average reward.
    """
    vec_env = make_environment()
    model = PPO.load("trained_models/swimmer")

    rewards = []
    for i in range(num_samples):
        obs = vec_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = vec_env.step(action)
            total_reward += reward
            if done:
                rewards.append(total_reward)

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return avg_reward, std_reward

if __name__=="__main__":
    # Must run with --train, --test, or --evaluate
    if len(sys.argv) != 2 or sys.argv[1] not in ["--train", "--test", "--eval"]:
        print("Usage: python swimmer.py [--train, --test, --eval]")
        sys.exit(1)

    if sys.argv[1] == "--train":
        train()
    elif sys.argv[1] == "--test":
        test()
    elif sys.argv[1] == "--eval":
        num_samples=10
        avg_reward, std_reward = evaluate(num_samples=num_samples)
        print(f"Reward over {num_samples} runs: {avg_reward} +/- {std_reward}")
