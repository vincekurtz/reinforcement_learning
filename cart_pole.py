#!/usr/bin/env python

##
#
# Train or test a policy with standard PPO on a simple cart-pole example. The
# goal is just to stabilize the cart-pole around the upright. No swing-up is
# required.
#
##

import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from policies import KoopmanPolicy
from envs import EnvWithObservationHistory

# Try to make things deterministic
set_random_seed(1, using_cuda=True)

def make_environment(render_mode=None):
    """
    Set up the gym environment (a.k.a. plant). Used for both training and
    testing.
    """
    return make_vec_env(EnvWithObservationHistory, n_envs=1,
                        env_kwargs={"env_name": "InvertedPendulum-v4", 
                                    "history_length": 1,
                                    "render_mode": render_mode})

def train():
    """
    Train the model with PPO and save it to disk.
    """
    vec_env = make_environment() 
    
    # set up the model (a.k.a. controller)
    model = PPO(KoopmanPolicy, vec_env, gamma=0.98, learning_rate=1e-3, 
                tensorboard_log="/tmp/cart_pole_tensorboard/",
                verbose=1)

    # Print how many parameters this thing has
    num_params = sum(p.numel() for p in model.policy.parameters())
    print("Training a policy with {num_params} parameters")

    # Do the learning
    model.learn(total_timesteps=200_000)

    # Save the model
    model.save("trained_models/cart_pole")

def test():
    """
    Load the trained model from disk and run a little simulation
    """
    vec_env = make_environment(render_mode="human")
    model = PPO.load("trained_models/cart_pole")

    obs = vec_env.reset()
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = vec_env.step(action)
        vec_env.render("human")

if __name__=="__main__":
    # Must run with --train or --test
    if len(sys.argv) != 2 or sys.argv[1] not in ["--train", "--test"]:
        print("Usage: python cart_pole.py [--train, --test]")
        sys.exit(1)

    if sys.argv[1] == "--train":
        train()
    elif sys.argv[1] == "--test":
        test()
