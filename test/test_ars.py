import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from brax.training import distribution
from brax.training.agents.ars.networks import ARSNetwork

from playground.architectures import MLP
from playground.ars import (
    BraxARSNetworkWrapper,
    make_policy_function,
    train_ars,
)
from playground.envs.pendulum.pendulum_env import PendulumSwingupEnv


def test_ars_wrapper():
    """Test the BraxARSNetworkWrapper."""
    observation_size = 3
    action_size = 2

    with pytest.raises(AssertionError):
        # We should get an error if the policy network's output doesn't match
        # the action size
        network_wrapper = BraxARSNetworkWrapper(
            policy_network=MLP(layer_sizes=(512, 3)),
        )
        network_wrapper.make_ars_network(
            observation_size=observation_size,
            action_size=action_size,
        )

    # We should end up with a PPONetworks object if everything is correct
    network_wrapper = BraxARSNetworkWrapper(
        policy_network=MLP(layer_sizes=(512, 2)),
    )
    ppo_networks = network_wrapper.make_ars_network(
        observation_size=observation_size,
        action_size=action_size,
    )
    assert isinstance(ppo_networks, ARSNetwork)


def test_ars_network_io():
    """Test saving and loading an ARSNetwork object."""
    observation_size = 3
    action_size = 2
    network_wrapper = BraxARSNetworkWrapper(
        policy_network=MLP(layer_sizes=(512, 2)),
    )
    ars_network = network_wrapper.make_ars_network(
        observation_size=observation_size,
        action_size=action_size,
    )
    assert isinstance(ars_network, ARSNetwork)

    # Save to a file
    local_dir = Path("_test_ars_networks_io")
    local_dir.mkdir(parents=True, exist_ok=True)
    model_path = local_dir / "ars_networks.pkl"
    with Path(model_path).open("wb") as f:
        pickle.dump(network_wrapper, f)

    # Load from a file and check that the network is the same
    with Path(model_path).open("rb") as f:
        new_network_wrapper = pickle.load(f)
    new_ars_network = new_network_wrapper.make_ars_network(
        observation_size=observation_size,
        action_size=action_size,
    )
    assert isinstance(new_ars_network, ARSNetwork)
    assert jax.tree_util.tree_structure(
        ars_network
    ) == jax.tree_util.tree_structure(new_ars_network)

    # Clean up
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


def test_train_ppo():
    """Test train wrapper for a simple PPO agent."""
    # Set up a random key
    rng = jax.random.PRNGKey(0)

    # Create a policy and value functions for a pendulum swingup task
    observation_size = 3
    action_size = 1
    network_wrapper = BraxARSNetworkWrapper(
        policy_network=MLP(layer_sizes=(16, 1)),
    )

    # Set up a temporary directory for saving the policy
    local_dir = Path("_test_train_ars")
    local_dir.mkdir(parents=True, exist_ok=True)
    save_path = local_dir / "pendulum_policy.pkl"

    # Train the agent
    _, params, make_policy = train_ars(
        env=PendulumSwingupEnv,
        network_wrapper=network_wrapper,
        save_path=save_path,
        tensorboard_logdir=local_dir,
        num_timesteps=20_000,
        num_evals=10,
        episode_length=100,
        normalize_observations=True,
        number_of_directions=60,
        top_directions=20,
        step_size=0.02,
        exploration_noise_std=0.03,
        seed=0,
    )

    # Run a forward pass through the trained policy
    policy = make_policy(params)

    # Check that the policy returns the correct action size
    obs_rng, act_rng = jax.random.split(rng)
    obs = jax.random.normal(obs_rng, (observation_size,))
    action, _ = policy(obs, act_rng)
    assert action.shape == (action_size,)

    # Load the trained policy from disk
    with Path(save_path).open("rb") as f:
        loaded_network_and_params = pickle.load(f)
    loaded_network_wrapper = loaded_network_and_params["network_wrapper"]
    loaded_params = loaded_network_and_params["params"]

    assert isinstance(loaded_network_wrapper, BraxARSNetworkWrapper)

    # Check that the loaded policy returns the same action
    loaded_policy = make_policy_function(
        loaded_network_wrapper,
        loaded_params,
        observation_size,
        action_size,
        normalize_observations=True,
    )
    new_action, _ = loaded_policy(obs, act_rng)
    assert jnp.allclose(action, new_action)

    # Clean up
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


if __name__ == "__main__":
    test_ars_wrapper()
    test_ars_network_io()
    test_train_ppo()
