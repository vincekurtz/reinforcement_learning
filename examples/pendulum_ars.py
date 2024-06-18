import pickle
import sys

import jax
from brax.training.distribution import NormalTanhDistribution

from playground.architectures import MLP
from playground.ars import (
    BraxARSNetworkWrapper,
    make_policy_function,
    train_ars,
)
from playground.envs.pendulum.pendulum_env import PendulumSwingupEnv
from playground.simulation import run_interactive, save_video

"""
Use ARS to train a pendulum swingup task.
"""


def train():
    """Train the swingup policy and save it to a file."""
    # Create policy and value functions
    network_wrapper = BraxARSNetworkWrapper(
        policy_network=MLP(layer_sizes=(8, 8, 1), activate_final=True),
    )

    # Train the policy
    train_ars(
        env=PendulumSwingupEnv,
        network_wrapper=network_wrapper,
        save_path="/tmp/pendulum_ars.pkl",
        tensorboard_logdir="/tmp/rl_playground/pendulum_ars",
        num_timesteps=200_000_000,
        num_evals=10,
        episode_length=100,
        number_of_directions=512,
        top_directions=128,
        step_size=0.01,
        exploration_noise_std=0.1,
        normalize_observations=True,
        reward_shift=0.0,
        seed=0,
    )


def test(interactive=True):
    """Test the swingup policy with an interactive mujoco simulation."""
    env = PendulumSwingupEnv()

    # Load the trained policy
    with open("/tmp/pendulum_ars.pkl", "rb") as f:
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
    )
    jit_policy = jax.jit(lambda obs: policy(obs, jax.random.PRNGKey(0))[0])

    # Run the sim
    if interactive:
        run_interactive(env, jit_policy, fixed_camera_id=0)
    else:
        save_video(env, jit_policy, "pendulum.mp4", camera_name="camera")


if __name__ == "__main__":
    usage_message = "Usage: python pendulum_ars.py [train|test|video]"

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
