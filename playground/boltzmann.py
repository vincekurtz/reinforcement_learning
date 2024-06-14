import flax.linen as nn
import jax
import jax.flatten_util
import jax.numpy as jnp
from brax.envs.base import PipelineEnv
from flax import struct


@struct.dataclass
class BoltzmannPolicySearchOptions:
    """Hyperparameters for Boltzmann policy search.

    episode_length: The number of timesteps in each episode.
    num_envs: The number of parameter samples to take (each in a separate env).
    temperature: The temperature parameter λ in the update rule.
    sigma: The standard deviation of the parameter noise.
    """

    episode_length: int
    num_envs: int
    temperature: float
    sigma: float


class BoltzmannPolicySearch:
    """A simple RL algorithm inspired by the MPPI update rule.

    The basic idea is to update the parameters of a feedback policy

        u = π(y; θ)

    via the MPPI-like update rule

        θᵢ ~ 𝒩(θ, σ²)
        θ ← ∑ᵢ θᵢ exp(R(θᵢ)/λ) / ∑ᵢ exp(R(θᵢ)/λ)

    where R(θ) is the total reward from a policy rollout, and λ > 0 is a
    temperature parameter.
    """

    def __init__(
        self,
        env: PipelineEnv,
        policy: nn.Module,
        options: BoltzmannPolicySearchOptions,
        seed: int = 0,
    ):
        """Initialize the Boltzmann policy search algorithm.

        Args:
            env: The environment to train on.
            policy: The policy module to train, maps observations to actions.
            options: The hyperparameters for the algorithm.
            seed: The random seed to use for parameter initialization.
        """
        self.env = env
        self.policy = policy
        self.options = options

        # Initialize the policy parameters
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((1, env.observation_size))
        init_params = self.policy.init(init_rng, dummy_obs)

        # Check that the policy outputs the correct action size
        dummy_out = self.policy.apply(init_params, dummy_obs)
        assert dummy_out.shape[-1] == env.action_size, (
            f"policy output size {dummy_out.shape[-1]} does not match "
            f"action size {env.action_size}"
        )

        # Make an unravelling function so we can treat params as a vector
        flat_params, self.unravel = jax.flatten_util.ravel_pytree(init_params)
        self.num_params = len(flat_params)
        self.initial_parameter_vector = flat_params

    def rollout(self, parameter_vector: jnp.ndarray, rng: jax.random.PRNGKey):
        """Roll out the policy with a given set of parameters.

        Args:
            parameter_vector: The parameters to use for the policy.
            rng: The random key to use for the rollout.

        Returns:
            The total reward from the rollout.
        """
        # Reset the environment
        start_state = self.env.reset(rng)
        params = self.unravel(parameter_vector)

        # Roll out the policy
        def f(carry, t):
            """Take a single step of the policy."""
            state, total_reward = carry
            action = self.policy.apply(params, state.obs)
            state = self.env.step(state, action)
            total_reward += state.reward
            return (state, total_reward), None

        (_, total_reward), _ = jax.lax.scan(
            f, (start_state, 0.0), jnp.arange(self.options.episode_length)
        )

        return total_reward

    def train(self, iterations: int, seed: int = 0):
        """Run the main training loop.

        Args:
            iterations: The number of training iterations to run.
            seed: The random seed to use for training.
        """
        rng = jax.random.PRNGKey(seed)
        jit_rollout = jax.jit(jax.vmap(self.rollout))
        param_vector = self.initial_parameter_vector

        for i in range(iterations):
            # Set random seeds for each rollout
            rng, rollout_rng = jax.random.split(rng)
            rollout_rng = jax.random.split(rollout_rng, self.options.num_envs)

            # Sample random normal perturbations to the parameters
            rng, param_rng = jax.random.split(rng)
            deltas = jax.random.normal(
                param_rng, (self.options.num_envs, self.num_params)
            )
            perturbed_params = param_vector + self.options.sigma * deltas

            # Roll out the perturbed policies
            rewards = jit_rollout(perturbed_params, rollout_rng)

            # Normalize the rewards
            mean_reward = jnp.mean(rewards)
            std_reward = jnp.std(rewards)
            rewards = (rewards - mean_reward) / std_reward

            # Compute the parameter update
            weights = jnp.exp(rewards / self.options.temperature)
            weights /= jnp.sum(weights)
            param_vector = jnp.sum(perturbed_params.T * weights, axis=1)

            if i % 50 == 0:
                print(f"Iteration {i}, mean reward: {mean_reward}")

        return self.unravel(param_vector)
