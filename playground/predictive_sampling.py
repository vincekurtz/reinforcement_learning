from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from brax.envs.base import PipelineEnv, State
from brax.training.types import Params
from flax import struct


@struct.dataclass
class PredictiveSamplingOptions:
    """Hyperparameters for predictive sampling policy search.

    episode_length: The number of timesteps in each episode.
    planning_horizon: The number of timesteps to plan ahead.
    num_envs: The number of parallel environments to use.
    num_samples: The number of samples to take in each environment.
    noise_std: The standard deviation of the noise added to actions.
    """

    episode_length: int
    planning_horizon: int
    num_envs: int
    num_samples: int
    noise_std: float


class PredictiveSampling:
    """Policy learning based on predictive sampling.

    The basic idea is to learn a trajectory-optimizing policy

        u₀, u₁, ... = π(y₀; θ)

    by regressing on training data generated by predictive sampling
    (Howell et al., https://arxiv.org/abs/2212.00541).
    """

    def __init__(
        self,
        env: PipelineEnv,
        policy: nn.Module,
        options: PredictiveSamplingOptions,
        seed: int = 0,
    ):
        """Initialize the predictive sampling policy search algorithm.

        Args:
            env: The environment to train on.
            policy: A network module mapping observations to an action sequence.
            options: The hyperparameters for the algorithm.
            seed: The random seed to use for parameter initialization.
        """
        self.env = env
        self.policy = policy
        self.options = options
        self.seed = seed

        # Initialize the policy parameters
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((1, env.observation_size))
        self.init_params = self.policy.init(init_rng, dummy_obs)

        # Check the policy has the correct output size
        dummy_output = self.policy.apply(self.init_params, dummy_obs)
        assert dummy_output.shape[-1] == (
            env.action_size * options.planning_horizon
        ), (
            f"Policy output size {dummy_output.shape[-1]} "
            f"does not match action sequence size "
            f"{env.action_size}x{options.planning_horizon} "
            f"= {env.action_size * options.planning_horizon}"
        )

    def rollout(
        self, start_state: State, action_sequence: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply the given action sequence and return the total reward.

        Args:
            start_state: The initial state of the environment.
            action_sequence: A sequence of actions to execute.

        Returns:
            The total reward from the rollout.
        """

        def step(carry, i):
            """Take a single step in the environment and sum the reward."""
            state, reward = carry
            action = action_sequence[i]
            state = self.env.step(state, action)
            reward += state.reward
            return (state, reward), None

        (_, total_reward), _ = jax.lax.scan(
            step, (start_state, 0.0), jnp.arange(self.options.planning_horizon)
        )
        return total_reward

    def choose_action_sequence(
        self,
        start_state: State,
        last_action_sequence: jnp.ndarray,
        policy_params: Params,
        rng: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Use predictive sampling to get a reasonable action sequence.

        Half of the samples are from a normal distribution around the last
        action sequence, while the other half are from a normal distribution
        around the policy output.

        Args:
            start_state: The initial state of the environment.
            last_action_sequence: The last action sequence executed.
            policy_params: The parameters of the policy.
            rng: The random key to use.

        Returns:
            The sampled action sequence with the highest reward.
        """
        rng, last_rng, policy_rng = jax.random.split(rng, 3)

        # Sample around the last action sequence
        mu_last = jnp.roll(last_action_sequence, -1, axis=0)
        mu_last = mu_last.at[-1].set(mu_last[-2])
        samples_from_last = (
            mu_last
            + self.options.noise_std
            * jax.random.normal(
                last_rng, (self.options.num_samples,) + mu_last.shape
            )
        )

        # Sample around the policy output as well
        mu_policy = self.policy.apply(policy_params, start_state.obs)
        mu_policy = jnp.reshape(mu_policy, (self.options.planning_horizon, -1))
        samples_from_policy = (
            mu_policy
            + self.options.noise_std
            * jax.random.normal(
                policy_rng, (self.options.num_samples,) + mu_policy.shape
            )
        )

        all_samples = jnp.concatenate([samples_from_last, samples_from_policy])

        # Roll out each action sequence and return the best one
        rewards = jax.vmap(self.rollout, in_axes=(None, 0))(
            start_state, all_samples
        )
        best_index = jnp.argmax(rewards)
        best_action_sequence = all_samples[best_index]
        return best_action_sequence

    def episode(
        self,
        policy_params: Params,
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Collect an episode of training data from a random initial state.

        Args:
            policy_params: The current policy parameters
            rng: The random key to use.

        Returns:
            A dataset of (start_state, action_sequence) pairs.
        """
        # Set a random initial state
        rng, reset_rng = jax.random.split(rng)
        state = self.env.reset(reset_rng)

        # Set a random initial action sequence
        rng, action_rng = jax.random.split(rng)
        action_sequence = self.options.noise_std * jax.random.normal(
            action_rng,
            (self.options.planning_horizon, self.env.action_size),
        )

        def f(carry, t):
            """Choose an action sequence and execute the first action."""
            start_state, last_action_sequence, rng = carry
            rng, sample_rng = jax.random.split(rng)

            action_sequence = self.choose_action_sequence(
                start_state, last_action_sequence, policy_params, sample_rng
            )
            state = self.env.step(start_state, action_sequence[0])
            return (state, action_sequence, rng), (
                start_state.obs,
                action_sequence,
            )

        (state, _, _), dataset = jax.lax.scan(
            f,
            (state, action_sequence, rng),
            jnp.arange(self.options.episode_length),
        )

        return dataset
