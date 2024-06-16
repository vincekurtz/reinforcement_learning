import jax

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
        episode_length=100,
        planning_horizon=10,
        num_envs=4,
        num_samples=8,
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


if __name__ == "__main__":
    test_rollout()
