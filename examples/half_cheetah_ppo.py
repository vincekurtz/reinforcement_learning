import pickle
import sys

import jax
from brax.training.distribution import NormalTanhDistribution

from playground.architectures import MLP
from playground.envs.half_cheetah.half_cheetah_env import HalfCheetahEnv
from playground.ppo import (
    BraxPPONetworksWrapper,
    make_policy_function,
    train_ppo,
)
from playground.simulation import run_interactive

"""
Use standard PPO to train a half-cheetah running task
"""


def train():
    """Train the policy and save it to a file."""
    # Create policy and value functions
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=MLP(layer_sizes=(16, 16, 12)),
        value_network=MLP(layer_sizes=(64, 64, 1)),
        action_distribution=NormalTanhDistribution,
    )

    # Train the policy
    train_ppo(
        env=HalfCheetahEnv,
        network_wrapper=network_wrapper,
        save_path="/tmp/half_cheetah_ppo.pkl",
        tensorboard_logdir="/tmp/rl_playground/half_cheetah_ppo",
        num_timesteps=80_000_000,
        num_evals=10,
        reward_scaling=0.1,
        episode_length=100,
        normalize_observations=True,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=1e-3,
        clipping_epsilon=0.2,
        entropy_cost=1e-3,
        num_envs=2048,
        batch_size=1024,
        seed=0,
    )


def test():
    """Test the policy with an interactive mujoco simulation."""
    env = HalfCheetahEnv()

    # Load the trained policy
    with open("/tmp/half_cheetah_ppo.pkl", "rb") as f:
        network_and_params = pickle.load(f)
    network_wrapper = network_and_params["network_wrapper"]
    params = network_and_params["params"]

    # Create a policy function
    policy = make_policy_function(
        network_wrapper=network_wrapper,
        params=params,
        observation_size=env.observation_size,
        action_size=env.action_size,
        normalize_observations=True,
        deterministic=True,
    )
    jit_policy = jax.jit(lambda obs: policy(obs, jax.random.PRNGKey(0))[0])

    # Run the sim
    run_interactive(env, jit_policy, fixed_camera_id=0)


if __name__ == "__main__":
    usage_message = "Usage: python half_cheetah_ppo.py [train|test]"

    if len(sys.argv) != 2:
        print(usage_message)
        sys.exit(1)

    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        test()
    else:
        print(usage_message)
        sys.exit(1)
