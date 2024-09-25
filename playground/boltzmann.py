from datetime import datetime
from typing import Tuple

import flax.linen as nn
import jax
import jax.flatten_util
import jax.numpy as jnp
from brax.envs.base import PipelineEnv
from brax.training.types import Metrics, Params
from flax import struct
from tensorboardX import SummaryWriter


@struct.dataclass
class BoltzmannPolicySearchOptions:
    """Hyperparameters for Boltzmann policy search.

    episode_length: The number of timesteps in each episode.
    num_envs: The number of parameter samples to take (each in a separate env).
    temperature: The temperature parameter λ in the update rule.
    sigma: The standard deviation of the parameter noise.
    num_eval_envs: The number of environments to use for evaluation.
    """

    episode_length: int
    num_envs: int
    temperature: float
    sigma: float
    num_eval_envs: int = 128


@struct.dataclass
class TrainingState:
    """Learned parameters and other training state data.

    param_vector: The current parameters of the policy.
    covariance_matrix: The covariance of the parameter sampling distribution.
    """

    param_vector: jnp.ndarray
    covariance_matrix: jnp.ndarray


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
        tensorboard_logdir: str = None,
    ):
        """Initialize the Boltzmann policy search algorithm.

        Args:
            env: The environment to train on.
            policy: The policy module to train, maps observations to actions.
            options: The hyperparameters for the algorithm.
            tensorboard_logdir: The directory to save tensorboard logs to.
        """
        self.env = env
        self.policy = policy
        self.options = options

        # Initialize the policy parameters
        rng = jax.random.PRNGKey(0)
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

        # Set up tensorboard logging
        if tensorboard_logdir is not None:
            logdir = tensorboard_logdir
        else:
            logdir = f"/tmp/rl_playground/bps_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print("Setting up tensorboard logging at", logdir)
        self.tb_writer = SummaryWriter(logdir)

    def make_training_state(self, rng: jax.random.PRNGKey) -> TrainingState:
        """Create a new set of initial training parameters.

        Args:
            rng: The random key to use for initialization.

        Returns:
            The initial training state.
        """
        dummy_obs = jnp.zeros((1, self.env.observation_size))
        params = self.policy.init(rng, dummy_obs)
        param_vector, _ = jax.flatten_util.ravel_pytree(params)
        covariance_matrix = jnp.eye(self.num_params) * self.options.sigma**2

        return TrainingState(param_vector, covariance_matrix)

    def evaluate(
        self, param_vector: jnp.ndarray, rng: jax.random.PRNGKey
    ) -> Metrics:
        """Evaluate the policy with a given set of parameters.

        Args:
            param_vector: The parameters to use for the policy.
            rng: The random seed to use for evaluation.

        Returns:
            A dictionary of evaluation info.
        """
        rng, eval_rng = jax.random.split(rng)
        eval_rng = jax.random.split(eval_rng, self.options.num_eval_envs)

        rewards, rollout_metrics = jax.vmap(self.rollout, in_axes=(None, 0))(
            param_vector, eval_rng
        )

        # Sum the rollout metrics over each episode
        rollout_metrics = jax.tree.map(
            lambda x: jnp.sum(
                x.reshape((self.options.num_eval_envs, -1)), axis=-1
            ),
            rollout_metrics,
        )

        metrics = {
            "eval/episode_reward": jnp.mean(rewards),
            "eval/episode_reward_std": jnp.std(rewards),
        }

        for key, val in rollout_metrics.items():
            metrics.update(
                {
                    f"eval/episode_{key}": jnp.mean(val),
                    f"eval/episode_{key}_std": jnp.std(val),
                }
            )

        return metrics

    def log_eval_data(self, info: dict, iteration: int) -> None:
        """Log evaluation info to tensorboard.

        Args:
            info: A dictionary of evaluation info.
            iteration: The current training iteration.
        """
        steps = iteration * self.options.num_envs * self.options.episode_length

        # Write stuff to tensorboard
        for key, value in info.items():
            if isinstance(value, jax.Array):
                value = float(value)
            self.tb_writer.add_scalar(key, value, steps)
        self.tb_writer.flush()

        # Print a summary
        elapsed = datetime.now() - self.start_time
        reward = info["eval/episode_reward"]
        print(f"  Steps: {steps}, Reward: {reward:.2f}, Time: {elapsed}")

    def rollout(
        self, param_vector: jnp.ndarray, rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, Metrics]:
        """Roll out the policy with a given set of parameters.

        Args:
            param_vector: The parameters to use for the policy.
            rng: The random key to use for the rollout.

        Returns:
            The total reward from the rollout.
        """
        # Reset the environment
        start_state = self.env.reset(rng)
        params = self.unravel(param_vector)

        # Roll out the policy
        def f(carry, t):
            """Take a single step of the policy."""
            state, total_reward = carry
            action = self.policy.apply(params, state.obs)
            state = self.env.step(state, action)
            total_reward += state.reward
            return (state, total_reward), state.metrics

        (_, total_reward), metrics = jax.lax.scan(
            f, (start_state, 0.0), jnp.arange(self.options.episode_length)
        )

        return total_reward, metrics

    def train_step(
        self, training_state: TrainingState, rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, Metrics]:
        """Perform a single step of training.

        Args:
            training_state: Container for the policy parameters.
            rng: The random key to use for the rollouts and sampling.

        Returns:
            The updated parameter vector, and a dictionary of training info.
        """
        param_vector = training_state.param_vector
        Sigma = training_state.covariance_matrix

        # Set random seeds for each rollout
        rng, rollout_rng = jax.random.split(rng)
        rollout_rng = jax.random.split(rollout_rng, self.options.num_envs)

        # Sample random normal perturbations to the parameters
        rng, param_rng = jax.random.split(rng)
        deltas = jax.random.multivariate_normal(
            param_rng,
            jnp.zeros(self.num_params),
            Sigma,
            shape=(self.options.num_envs,),
        )
        perturbed_params = param_vector + deltas

        # Roll out the perturbed policies
        rewards, _ = jax.vmap(self.rollout)(perturbed_params, rollout_rng)

        # Normalize the rewards
        mean_reward = jnp.mean(rewards)
        std_reward = jnp.std(rewards)
        rewards = (rewards - mean_reward) / (std_reward + 1e-6)

        # Compute the parameter update
        weights = jax.nn.softmax(rewards / self.options.temperature)
        param_vector = jnp.sum(perturbed_params.T * weights, axis=1)

        training_state = training_state.replace(
            param_vector=param_vector,
            # covariance_matrix=0.9 * Sigma + 0.1 * sample_covariance,
        )
        info = {
            "training/mean_reward": mean_reward,
            "training/std_reward": std_reward,
            "training/covariance_trace": jnp.trace(
                training_state.covariance_matrix
            ),
        }
        return training_state, info

    def train(self, iterations: int, num_evals: int, seed: int = 0) -> Params:
        """Run the main training loop.

        Args:
            iterations: The number of training iterations to run.
            num_evals: The number of times to print stuff out.
            seed: The random seed to use for training.

        Returns:
            The final parameters of the policy, in pytree format.
        """
        rng = jax.random.PRNGKey(seed)

        rng, init_rng = jax.random.split(rng)
        training_state = self.make_training_state(init_rng)

        jit_train_step = jax.jit(self.train_step)
        jit_evaluate = jax.jit(self.evaluate)

        self.start_time = datetime.now()
        eval_every = iterations // num_evals
        for i in range(iterations):
            rng, train_rng = jax.random.split(rng)
            training_state, train_metrics = jit_train_step(
                training_state, train_rng
            )

            if i % eval_every == 0:
                rng, eval_rng = jax.random.split(rng)
                metrics = jit_evaluate(training_state.param_vector, eval_rng)
                metrics.update(train_metrics)
                self.log_eval_data(metrics, i)

        return self.unravel(training_state.param_vector)
