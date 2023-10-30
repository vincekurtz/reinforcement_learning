#!/usr/bin/env python

##
#
# Train or test a policy with standard PPO on the half-cheetah gym example
#
##

import sys
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from torch import nn

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from policies import KoopmanPolicy
from envs import HistoryWrapper
import gymnasium as gym

# Whether to run the baseline MLP implementation from stable-baselines3 rl zoo
MLP_BASELINE = False

# Try to make things deterministic
SEED = 1
set_random_seed(SEED, using_cuda=True)

def make_environment(render_mode=None):
    """
    Set up the gym environment (a.k.a. plant). Used for both training and
    testing.
    """
    env = gym.make("HalfCheetah-v4", render_mode=render_mode)
    env.action_space.seed(SEED)
    if not MLP_BASELINE:
        env = HistoryWrapper(env, 10)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env.seed(SEED)

    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    return vec_env

def train():
    """
    Train the model with PPO and save it to disk.
    """
    vec_env = make_environment() 
    
    # set up the model (a.k.a. controller)
    if MLP_BASELINE:
        # Use tuned hyperparameters from RL ZOO, https://huggingface.co/sb3/ppo-HalfCheetah-v3
        model = PPO('MlpPolicy', vec_env, verbose=1, 
                    batch_size=64, clip_range=0.1, ent_coef=0.000401762,
                    gae_lambda=0.92, gamma=0.98, learning_rate=2.0633e-5, 
                    max_grad_norm=0.8, n_epochs=20, n_steps=512,
                    policy_kwargs=dict(log_std_init=-2, ortho_init=False, 
                                   activation_fn=nn.ReLU,
                                   net_arch=dict(pi=[256, 256], vf=[256, 256])),
                    vf_coef=0.58096,
                    tensorboard_log="/tmp/half_cheetah_tensorboard/")
    else:
        model = PPO(KoopmanPolicy, vec_env, verbose=1,
                    batch_size=64, clip_range=0.1, ent_coef=0.000401762,
                    gae_lambda=0.92, gamma=0.98, learning_rate=2.0633e-5, 
                    max_grad_norm=0.8, n_epochs=20, n_steps=512, vf_coef=0.58096,
                    policy_kwargs={"lifting_dim": 256,
                                   "log_std_init": -2,
                                   "ortho_init": False},
                    tensorboard_log="/tmp/half_cheetah_tensorboard/")

    # Print how many parameters this thing has
    num_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Training a policy with {num_params} parameters")
    print(model.policy)

    # Do the learning
    model.learn(total_timesteps=1_000_000)

    # Save the model
    model.save("trained_models/half_cheetah")

def test():
    """
    Load the trained model from disk and run a little simulation
    """
    vec_env = make_environment(render_mode="human")
    model = PPO.load("trained_models/half_cheetah")

    obs = vec_env.reset()
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = vec_env.step(action)
        vec_env.render("human")

if __name__=="__main__":
    # Must run with --train or --test
    if len(sys.argv) != 2 or sys.argv[1] not in ["--train", "--test"]:
        print("Usage: python half_cheetah.py [--train, --test]")
        sys.exit(1)

    if sys.argv[1] == "--train":
        train()
    elif sys.argv[1] == "--test":
        test()
