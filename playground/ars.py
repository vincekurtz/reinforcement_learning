import pickle
from datetime import datetime

import flax.linen as nn
import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.base import PipelineEnv
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.ars import train as ars
from brax.training.agents.ars.networks import ARSNetwork, make_inference_fn
from brax.training.types import Params
from flax import struct
from tensorboardX import SummaryWriter

"""
Interface for Brax's Augmented Random Search (ARS) implementation.
"""


@struct.dataclass
class BraxARSNetworkWrapper:
    """A lightweight wrapper around brax's ARSNetwork.

    Allows us to more easily save and load networks with non-default architectures.
    """

    policy_network: nn.Module

    def make_ars_network(
        self,
        observation_size: int,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        check_sizes: bool = True,
    ) -> ARSNetwork:
        """Create an ARSNetwork object, compatible with brax's ars.train() function.

        Args:
            observation_size: Size of the input (observation).
            action_size: Size of the policy output (action).
            preprocess_observations_fn: Function to preprocess (e.g. normalize) observations.
            check_sizes: Whether to check that the output sizes of the policy and value networks match the action and value distributions.

        Returns:
            An ARSNetwork object.
        """
        # Set up a dummy observation for parameter initialization.
        dummy_observation = jnp.zeros((1, observation_size))

        if check_sizes:
            rng = jax.random.PRNGKey(0)
            params = self.policy_network.init(rng, dummy_observation)
            action = self.policy_network.apply(params, dummy_observation)
            assert action.shape == (
                1,
                action_size,
            ), f"Action shape is {action.shape}, expected {(1, action_size)}"

        def policy_init(key):
            """Initialize the policy parameters from a random key."""
            return self.policy_network.init(key, dummy_observation)

        def policy_apply(processor_params, policy_params, obs):
            """Compute actions from observations."""
            obs = preprocess_observations_fn(obs, processor_params)
            return self.policy_network.apply(policy_params, obs)

        return ARSNetwork(init=policy_init, apply=policy_apply)


def train_ars(
    env: PipelineEnv,
    network_wrapper: BraxARSNetworkWrapper,
    save_path: str = None,
    tensorboard_logdir: str = None,
    **kwargs,
):
    """Train an ARS agent and save the learned policy parameters.

    Args:
        env: The environment to train on.
        network_wrapper: A BraxARSNetworkWrapper object that contains the policy network.
        save_path: Path to save the trained policy.
        tensorboard_logdir: Path to save TensorBoard logs.
        **kwargs: Additional arguments to pass to ars.train().

    Returns:
        A dictionary of training metrics.
        The trained policy parameters
        A function to make policies from these parameters.
    """
    # Initilize the environment
    print("Initializing environment...")
    envs.register_environment("ars_training_env", env)
    env = envs.get_environment("ars_training_env")

    # A separate eval env is required for domain randomization
    eval_env = envs.get_environment("ars_training_env")

    # Define a tensorboard logging callback
    if tensorboard_logdir is not None:
        logdir = tensorboard_logdir
    else:
        logdir = (
            f"/tmp/rl_playground/ars_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    print(f"Setting up TensorBoard logging in {logdir}...")

    writer = SummaryWriter(logdir)
    times = [datetime.now()]

    def progress(step, metrics):
        print(f"  Steps: {step}, Reward: {metrics['eval/episode_reward']}")
        times.append(datetime.now())

        # Write all metrics to tensorboard
        for key, val in metrics.items():
            if isinstance(val, jax.Array):
                val = float(val)  # we need floats for logging
            writer.add_scalar(key, val, step)

    # Train the agent
    print("Training agent...")
    make_policy, params, metrics = ars.train(
        environment=env,
        progress_fn=progress,
        network_factory=network_wrapper.make_ars_network,
        eval_env=eval_env,
        **kwargs,
    )

    print(f"Time to jit: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

    # Save the trained policy to disk
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "network_wrapper": network_wrapper,
                    "params": params,
                },
                f,
            )

    print("Done!")
    return metrics, params, make_policy


def make_policy_function(
    network_wrapper: BraxARSNetworkWrapper,
    params: Params,
    observation_size: int,
    action_size: int,
    normalize_observations: bool = True,
):
    """Create a policy function from a trained ARS network.

    Args:
        network_wrapper: A BraxARSNetworkWrapper object that contains the policy network.
        params: The trained policy parameters.
        observation_size: Size of the policy input (observation).
        action_size: Size of the policy output (action).
        normalize_observations: Whether to normalize observations.

    Returns:
        A function that takes observations and returns actions.
    """
    if normalize_observations:
        preprocess_observations_fn = running_statistics.normalize
    else:
        preprocess_observations_fn = types.identity_observation_preprocessor

    ars_network = network_wrapper.make_ars_network(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
    )
    make_policy = make_inference_fn(ars_network)
    policy = make_policy(params)
    return policy
