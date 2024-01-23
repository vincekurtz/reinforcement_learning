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
import torch

from policies import KoopmanPolicy

# Whether to use a standard MLP as a baseline
MLP_BASELINE = False

if MLP_BASELINE:
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
else:
    from sb3_mod import PPO
    from sb3_mod.common.utils import set_random_seed
    from sb3_mod.common.vec_env import DummyVecEnv, VecNormalize
    from sb3_mod.common.monitor import Monitor

# Try to make things deterministic
SEED = 1
set_random_seed(SEED, using_cuda=True)

def make_environment(render_mode=None):
    """
    Set up the gym environment (a.k.a. plant). Used for both training and
    testing.
    """
    env = gym.make("HumanoidStandup-v4", render_mode=render_mode)
    env.action_space.seed(SEED)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env.seed(SEED)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    return vec_env

def train():
    """
    Train the model with PPO and save it to disk.
    """
    vec_env = make_environment() 
    
    if MLP_BASELINE:
        model = PPO("MlpPolicy", vec_env, 
                    batch_size=32,
                    n_steps=512,
                    gamma=0.99,
                    learning_rate=2.6e-5,
                    ent_coef=3.62e-6,
                    clip_range=0.3,
                    n_epochs=20,
                    gae_lambda=0.9,
                    max_grad_norm=0.7,
                    vf_coef=0.43,
                    policy_kwargs=dict(
                        log_std_init=-2,
                        ortho_init=False,
                        net_arch=dict(pi=[256, 256], vf=[256, 256], 
                                      activation_fn=torch.nn.GELU)),
                    tensorboard_log="/tmp/humanoid_tensorboard/",
                    verbose=1)
    else:
        model = PPO(KoopmanPolicy, vec_env, 
                    batch_size=32,
                    n_steps=512,
                    gamma=0.99,
                    learning_rate=3e-5,
                    ent_coef=1e-4,
                    clip_range=0.3,
                    n_epochs=20,
                    gae_lambda=0.9,
                    max_grad_norm=0.7,
                    vf_coef=0.5,
                    tensorboard_log="/tmp/humanoid_tensorboard/",
                    koopman_coef=1.0,
                    verbose=1, policy_kwargs={"lifting_dim": 256})

    # Print how many parameters this thing has
    num_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Training a policy with {num_params} parameters")
    print(model.policy)

    # Do the learning
    model.learn(total_timesteps=1_000_000)

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

