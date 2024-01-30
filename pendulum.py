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

# Whether to run the baseline MLP implementation from stable-baselines3 rl zoo
MLP_BASELINE = True

if MLP_BASELINE:
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
else:
    from sb3_mod import PPO
    from sb3_mod.common.utils import set_random_seed
    from sb3_mod.common.vec_env import DummyVecEnv
    from sb3_mod.common.monitor import Monitor

import torch
import numpy as np

from policies import KoopmanPolicy
from envs import HistoryWrapper


# Try to make things deterministic
SEED = 2
set_random_seed(SEED, using_cuda=True)

def make_environment(render_mode=None):
    """
    Set up the gym environment (a.k.a. plant). Used for both training and
    testing.
    """
    max_torque = 2.0
    env = gym.make("Pendulum-v1", render_mode=render_mode, g=10.0)
    env.unwrapped.max_torque = max_torque
    env.unwrapped.action_space.low = -max_torque
    env.unwrapped.action_space.high = max_torque
    env.action_space.seed(SEED)
    if not MLP_BASELINE:
        env = HistoryWrapper(env, 1)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env.seed(SEED)
    return vec_env

def train():
    """
    Train the model with PPO and save it to disk.
    """
    vec_env = make_environment() 
    
    # set up the model (a.k.a. controller)
    if MLP_BASELINE:
        model = PPO("MlpPolicy", vec_env, gamma=0.98, learning_rate=1e-3,
                    tensorboard_log="/tmp/pendulum_tensorboard/",
                    policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.GELU),
                    verbose=1)
    else:
        model = PPO(KoopmanPolicy, vec_env, gamma=0.98, learning_rate=1e-3,
                    tensorboard_log="/tmp/pendulum_tensorboard/",
                    koopman_coef=10.0,
                    verbose=1, policy_kwargs={"lifting_dim": 256})

    # Print how many parameters this thing has
    num_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Training a policy with {num_params} parameters")
    print(model.policy)

    # Do the learning
    model.learn(total_timesteps=100_000)

    # Save the model
    model.save("trained_models/pendulum")

def test():
    """
    Load the trained model from disk and run a little simulation
    """
    vec_env = make_environment(render_mode="human")
    model = PPO.load("trained_models/pendulum")

    def predict_next_obs(obs):
        """Helper function to predict the next observation"""
        obs_th = torch.Tensor(obs).to(model.device)
        obs_pred_th = model.policy.mlp_extractor.predict_next_observation(obs_th)
        obs_pred = obs_pred_th.cpu().detach().numpy()
        return obs_pred

    obs = vec_env.reset()
    obs_pred = predict_next_obs(obs)
    for i in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = vec_env.step(action)
        vec_env.render("human")

        # Compute the error in predicting the next observation
        prediction_error = np.linalg.norm(obs - obs_pred)
        print(f"Prediction error: {prediction_error:.3f}")
        obs_pred = predict_next_obs(obs)


if __name__=="__main__":
    # Must run with --train or --test
    if len(sys.argv) != 2 or sys.argv[1] not in ["--train", "--test"]:
        print("Usage: python pendulum.py [--train, --test]")
        sys.exit(1)

    if sys.argv[1] == "--train":
        train()
    elif sys.argv[1] == "--test":
        test()
