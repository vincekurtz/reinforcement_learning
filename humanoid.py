#!/usr/bin/env python

##
#
# Train or test a policy with standard PPO on a simple inverted pendulum. The
# pendulum has low torque limits, so must pump energy into the system and then
# stabilize around the upright. 
#
##

import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from policies import KoopmanPolicy

# Whether to use a standard MLP as a baseline
MLP_BASELINE = True

# Try to make things deterministic
SEED = 1
set_random_seed(SEED, using_cuda=True)

def make_environment(render_mode=None):
    """
    Set up the gym environment (a.k.a. plant). Used for both training and
    testing.
    """
    env = gym.make("Humanoid-v4", render_mode=render_mode)
    env.action_space.seed(SEED)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env.seed(SEED)
    return vec_env

def train():
    """
    Train the model with PPO and save it to disk.
    """
    vec_env = make_environment() 
    
    if MLP_BASELINE:
        model = PPO("MlpPolicy", vec_env, 
                    policy_kwargs=dict(
                        net_arch=dict(pi=[1024,1024], vf=[1024,1024])),
                    tensorboard_log="/tmp/humanoid_tensorboard/",
                    verbose=1)
    else:
        model = PPO(KoopmanPolicy, vec_env, 
                    tensorboard_log="/tmp/humanoid_tensorboard/",
                    verbose=1, policy_kwargs={
                        "lifting_dim": 1024,
                        "num_layers": 3,
                        "ortho_init": False})

    # Print how many parameters this thing has
    num_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"Training a policy with {num_params} parameters")
    print(model.policy)

    # Do the learning
    model.learn(total_timesteps=90_000_000)

    # Save the model
    model.save("trained_models/humanoid")

def test():
    """
    Load the trained model from disk and run a little simulation
    """
    vec_env = make_environment(render_mode="rgb_array")
    model = PPO.load("trained_models/humanoid")
    
    num_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Loaded policy with {num_params} parameters")

    obs = vec_env.reset()
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = vec_env.step(action)
        vec_env.render("human")
        import time
        time.sleep(0.01)

if __name__=="__main__":
    # Must run with --train or --test
    if len(sys.argv) != 2 or sys.argv[1] not in ["--train", "--test"]:
        print("Usage: python humanoid.py [--train, --test]")
        sys.exit(1)

    if sys.argv[1] == "--train":
        train()
    elif sys.argv[1] == "--test":
        test()

