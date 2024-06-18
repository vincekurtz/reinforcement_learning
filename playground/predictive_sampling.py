from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from brax.envs.base import PipelineEnv, State
from brax.training.types import Params
from flax import struct
from optax import OptState


@struct.dataclass
class PredictiveSamplingOptions:
    """Hyperparameters for predictive sampling policy search.

    episode_length: The number of timesteps in each episode.
    planning_horizon: The number of timesteps to plan ahead.
    num_envs: The number of parallel environments to use.
    num_samples: The number of samples to take in each environment.
    noise_std: The standard deviation of the noise added to actions.
    learning_rate: The learning rate for policy regression
    batch_size: The number of samples to use in each training batch.
    epochs_per_iteration: The number of policy regression epochs per iteration.
    iterations: The number of iterations (sampling followed by regression)
    """

    episode_length: int
    planning_horizon: int
    num_envs: int
    num_samples: int
    noise_std: float
    learning_rate: float
    batch_size: int
    epochs_per_iteration: int
    iterations: int


@struct.dataclass
class TrainingState:
    """Learned parameters and optimizer state for policy training.

    params: The parameters of the policy network.
    opt_state: The optimizer (e.g., Adam) state.
    """

    params: Params
    opt_state: OptState


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
    ):
        """Initialize the predictive sampling policy search algorithm.

        Args:
            env: The environment to train on.
            policy: A network module mapping observations to an action sequence.
            options: The hyperparameters for the algorithm.
        """
        assert (
            options.num_envs * options.episode_length % options.batch_size == 0
        ), (
            f"The batch size {options.batch_size} must divide the number of "
            f"data points, {options.num_envs * options.episode_length}."
        )

        self.env = env
        self.policy = policy
        self.options = options
        self.optimizer = optax.adam(self.options.learning_rate)

    def make_training_state(self, rng: jax.random.PRNGKey) -> TrainingState:
        """Initialize all learnable parameters.

        Args:
            rng: The random key to use for parameter initialization.

        Returns:
            A container with all learnable parameters.
        """
        # Initialize the policy parameters
        rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((1, self.env.observation_size))
        init_params = self.policy.init(init_rng, dummy_obs)

        # Check the policy has the correct output size
        dummy_output = self.policy.apply(init_params, dummy_obs)
        assert dummy_output.shape[-1] == (
            self.env.action_size * self.options.planning_horizon
        ), (
            f"Policy output size {dummy_output.shape[-1]} "
            f"does not match action sequence size "
            f"{self.env.action_size}x{self.options.planning_horizon} "
            f"= {self.env.action_size * self.options.planning_horizon}"
        )

        # Initialize the optimizer state
        opt_state = self.optimizer.init(init_params)

        return TrainingState(params=init_params, opt_state=opt_state)

    def apply_policy(self, params: Params, obs: jnp.ndarray) -> jnp.ndarray:
        """Get a control action sequence u₀, u₁, ... = π(y₀; θ).

        Args:
            params: The parameters of the policy network θ.
            obs: The initial observation y₀.

        Returns:
            The action sequence u₀, u₁, ....
        """
        flat_actions = self.policy.apply(params, obs)
        return jnp.reshape(
            flat_actions,
            obs.shape[:-1]
            + (self.options.planning_horizon, self.env.action_size),
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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
            The reward associated with the best action sequence.
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
        mu_policy = self.apply_policy(policy_params, start_state.obs)
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
        return best_action_sequence, rewards[best_index]

    def episode(
        self,
        policy_params: Params,
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Collect an episode of training data from a random initial state.

        Args:
            policy_params: The current policy parameters
            rng: The random key to use.

        Returns:
            A dataset of (start_state, action_sequence, reward).
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

            action_sequence, reward = self.choose_action_sequence(
                start_state, last_action_sequence, policy_params, sample_rng
            )
            state = self.env.step(start_state, action_sequence[0])
            return (state, action_sequence, rng), (
                start_state.obs,
                action_sequence,
                reward,
            )

        (state, _, _), dataset = jax.lax.scan(
            f,
            (state, action_sequence, rng),
            jnp.arange(self.options.episode_length),
        )

        # TODO: log difference between policy and optimal action sequence

        return dataset

    def regress_policy(
        self,
        training_state: TrainingState,
        observations: jnp.ndarray,
        action_sequences: jnp.ndarray,
        rng: jax.random.PRNGKey,
    ) -> Tuple[TrainingState, jnp.ndarray]:
        """Fit the policy to the given observations and action sequences.

        Args:
            training_state: Parameters for the policy u₀, u₁, ... = π(y₀; θ).
            observations: Initial observations y₀.
            action_sequences: Action sequences u₀, u₁, ....
            rng: The random key to use for shuffling the data.

        Returns:
            Updated policy parameters and optimizer state.
            Training loss from the last epoch.
        """
        num_data_points = self.options.num_envs * self.options.episode_length
        num_batches = num_data_points // self.options.batch_size

        def _loss(params, obs, act):
            """Compute the mean squared error loss."""
            act_pred = self.apply_policy(params, obs)
            return jnp.mean(jnp.square(act - act_pred))

        loss_and_grad = jax.value_and_grad(_loss)

        def _batch_step(carry, idx):
            """Do a gradient descent step on a single batch."""
            params, opt_state, permutation, _ = carry

            # Select data from this batch
            batch_idx = jax.lax.dynamic_slice_in_dim(
                permutation,
                idx * self.options.batch_size,
                self.options.batch_size,
            )
            batch_obs = observations[batch_idx]
            batch_act = action_sequences[batch_idx]

            # Take the gradient step
            loss, grads = loss_and_grad(params, batch_obs, batch_act)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            return (params, opt_state, permutation, loss), None

        def _epoch(carry, i):
            """Do an epoch of training (pass over all training data once)."""
            params, opt_state, rng = carry

            # Shuffle the data
            rng, shuffle_rng = jax.random.split(rng)
            permutation = jax.random.permutation(shuffle_rng, num_data_points)

            # Do a gradient descent step on each batch
            (params, opt_state, _, loss), _ = jax.lax.scan(
                _batch_step,
                (params, opt_state, permutation, 0.0),
                jnp.arange(num_batches),
            )

            return (params, opt_state, rng), loss

        params, opt_state = training_state.params, training_state.opt_state
        rng, epoch_rng = jax.random.split(rng)
        (params, opt_state, _), losses = jax.lax.scan(
            _epoch,
            (params, opt_state, epoch_rng),
            jnp.arange(self.options.epochs_per_iteration),
        )

        loss = losses[-1]
        return training_state.replace(params=params, opt_state=opt_state), loss

    def train(self, seed=0) -> Params:
        """Main training loop for predictive sampling policy search.

        Args:
            seed: The random seed to use for training.

        Returns:
            The learned policy parameters.
        """
        rng = jax.random.PRNGKey(seed)

        # Choose some initial parameters
        rng, init_rng = jax.random.split(rng)
        training_state = self.make_training_state(init_rng)

        # Function to gather training data
        jit_episode = jax.jit(jax.vmap(self.episode, in_axes=(None, 0)))

        # Function to regress the policy
        jit_regress = jax.jit(self.regress_policy)

        for i in range(self.options.iterations):
            # Collect training data
            rng, episode_rng = jax.random.split(rng)
            episode_rngs = jax.random.split(episode_rng, self.options.num_envs)
            observations, action_sequences, rewards = jit_episode(
                training_state.params, episode_rngs
            )

            # Fit the policy to the training data
            rng, regression_rng = jax.random.split(rng)
            training_state, loss = jit_regress(
                training_state, observations, action_sequences, regression_rng
            )

            print(
                f"Iteration {i+1} complete, loss = {loss}, reward = {jnp.mean(rewards)}"
            )

        return training_state.params
