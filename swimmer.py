#!/usr/bin/env python

##
#
# Train or test a policy with standard PPO on a little swimmer robot.
#
##

import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib.common.wrappers import TimeFeatureWrapper

from policies import KoopmanPolicy
from envs import EnvWithObservationHistory

# Try to make things deterministic
set_random_seed(1, using_cuda=True)

def make_environment(render_mode=None):
    """
    Set up the gym environment (a.k.a. plant). Used for both training and
    testing.
    """
    #return make_vec_env(EnvWithObservationHistory, n_envs=1,
    #                    env_kwargs={"env_name": "Swimmer-v4", 
    #                                "history_length": 32,
    #                                "render_mode": render_mode})
    vec_env = make_vec_env("Swimmer-v4", n_envs=1,
                        wrapper_class=TimeFeatureWrapper,
                        env_kwargs={"render_mode": render_mode})
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    return vec_env

def train():
    """
    Train the model with PPO and save it to disk.
    """
    vec_env = make_environment() 
    
    # set up the model (a.k.a. controller)
    #model = PPO(KoopmanPolicy, vec_env, gamma=0.9999,
    #            tensorboard_log="/tmp/swimmer_tensorboard/",
    #            verbose=1, policy_kwargs={"num_linear_systems": 8})
    model = PPO('MlpPolicy', vec_env, verbose=1, gamma=0.9999,
                tensorboard_log="/tmp/swimmer_tensorboard/")

    # Print how many parameters this thing has
    num_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Training a policy with {num_params} parameters")

    # Do the learning
    model.learn(total_timesteps=1_000_000)

    # Save the model
    model.save("trained_models/swimmer")

def test():
    """
    Load the trained model from disk and run a little simulation
    """
    vec_env = make_environment(render_mode="human")
    model = PPO.load("trained_models/swimmer")

    obs = vec_env.reset()
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = vec_env.step(action)
        vec_env.render("human")

if __name__=="__main__":
    # Must run with --train or --test
    if len(sys.argv) != 2 or sys.argv[1] not in ["--train", "--test"]:
        print("Usage: python swimmer.py [--train, --test]")
        sys.exit(1)

    if sys.argv[1] == "--train":
        train()
    elif sys.argv[1] == "--test":
        test()
