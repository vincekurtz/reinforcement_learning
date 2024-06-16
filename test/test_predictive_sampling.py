import jax
import jax.numpy as jnp

from playground.architectures import MLP
from playground.envs.pendulum.pendulum_env import PendulumSwingupEnv
from playground.predictive_sampling import (
    PredictiveSampling,
    PredictiveSamplingOptions,
)


def make_optimizer():
    """Make a simple PredictiveSampling instance."""
    env = PendulumSwingupEnv()
    options = PredictiveSamplingOptions(
        episode_length=1000,
        planning_horizon=20,
        num_envs=4,
        num_samples=512,
        noise_std=0.1,
    )
    policy = MLP(layer_sizes=(8, 8, options.planning_horizon * env.action_size))
    return PredictiveSampling(env, policy, options)


def test_rollout():
    """Test rolling out an action sequence."""
    rng = jax.random.PRNGKey(0)
    ps = make_optimizer()
    jit_reset = jax.jit(ps.env.reset)
    jit_step = jax.jit(ps.env.step)

    rng, reset_rng, action_rng = jax.random.split(rng, 3)
    start_state = jit_reset(reset_rng)
    action_sequence = jax.random.normal(
        action_rng,
        (ps.options.planning_horizon, ps.env.action_size),
    )

    # Manually apply the action sequence
    manual_reward = 0.0
    state = start_state
    for action in action_sequence:
        state = jit_step(state, action)
        manual_reward += state.reward

    # Use the rollout method
    reward = ps.rollout(start_state, action_sequence)
    assert reward == manual_reward


def test_choose_action_sequence():
    """Test choosing an action sequence."""
    rng = jax.random.PRNGKey(0)
    ps = make_optimizer()
    jit_reset = jax.jit(ps.env.reset)

    rng, reset_rng, act_rng, sample_rng = jax.random.split(rng, 4)
    start_state = jit_reset(reset_rng)
    last_action_sequence = jax.random.normal(
        act_rng,
        (ps.options.planning_horizon, ps.env.action_size),
    )
    policy_params = None  # TODO: include policy

    best_action_sequence = ps.choose_action_sequence(
        start_state, last_action_sequence, policy_params, sample_rng
    )
    assert best_action_sequence.shape == (
        ps.options.planning_horizon,
        ps.env.action_size,
    )
    best_reward = ps.rollout(start_state, best_action_sequence)
    other_reward = ps.rollout(start_state, jnp.zeros_like(best_action_sequence))
    assert best_reward > other_reward


def test_episode():
    """Test running an episode from a single initial state."""
    rng = jax.random.PRNGKey(0)
    ps = make_optimizer()
    policy_params = None  # TODO: include policy

    rng, episode_rng = jax.random.split(rng)
    obs, actions = ps.episode(policy_params, episode_rng)
    print("last obs: ", obs[-1])


if __name__ == "__main__":
    # test_rollout()
    # test_choose_action_sequence()
    test_episode()
