import pickle
import sys

import jax
from brax.training.distribution import NormalTanhDistribution

from playground.architectures import MLP
from playground.envs.cart_pole.cart_pole_env import CartPoleSwingupEnv
from playground.ppo import (
    BraxPPONetworksWrapper,
    make_policy_function,
    train_ppo,
)
from playground.simulation import run_interactive

"""
Use standard PPO to train a cart-pole swingup task.
"""


def train():
    """Train the swingup policy and save it to a file."""
    # Create policy and value functions
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=MLP(layer_sizes=(32, 32, 2)),
        value_network=MLP(layer_sizes=(64, 64, 1)),
        action_distribution=NormalTanhDistribution,
    )

    # Train the policy
    train_ppo(
        env=CartPoleSwingupEnv,
        network_wrapper=network_wrapper,
        save_path="/tmp/cart_pole_ppo.pkl",
        tensorboard_logdir="/tmp/rl_playground/cart_pole_ppo",
        num_timesteps=60_000_000,
        num_evals=10,
        reward_scaling=0.01,
        episode_length=100,
        normalize_observations=True,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=1e-3,
        clipping_epsilon=0.2,
        entropy_cost=1e-2,
        num_envs=2048,
        batch_size=1024,
        seed=0,
    )


def test():
    """Test the swingup policy with an interactive mujoco simulation."""
    env = CartPoleSwingupEnv()

    # Load the trained policy
    with open("/tmp/cart_pole_ppo.pkl", "rb") as f:
        network_and_params = pickle.load(f)
    network_wrapper = network_and_params["network_wrapper"]
    params = network_and_params["params"]

    # Create a policy function
    policy = make_policy_function(
        network_wrapper=network_wrapper,
        params=params,
        observation_size=3,
        action_size=1,
        normalize_observations=True,
        deterministic=True,
    )
    jit_policy = jax.jit(lambda obs: policy(obs, jax.random.PRNGKey(0))[0])

    # Run the sim
    run_interactive(env, jit_policy, fixed_camera_id=0)


if __name__ == "__main__":
    usage_message = "Usage: python cart_pole_ppo.py [train|test]"

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
