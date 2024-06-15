import jax

from playground.architectures import MLP
from playground.boltzmann import (
    BoltzmannPolicySearch,
    BoltzmannPolicySearchOptions,
)
from playground.envs.pendulum.pendulum_env import PendulumSwingupEnv


def test_bps():
    """Test Boltzmann Policy Search."""
    rng = jax.random.PRNGKey(0)
    env = PendulumSwingupEnv()
    policy = MLP(layer_sizes=(8, 8, 1))

    options = BoltzmannPolicySearchOptions(
        episode_length=100,
        num_envs=32,
        temperature=1.0,
        sigma=0.1,
    )
    bps = BoltzmannPolicySearch(env, policy, options)
    assert bps.num_params > 0

    # Check that a single rollout works
    rng, rollout_rng = jax.random.split(rng)
    param_vec = bps.initial_parameter_vector
    reward, _ = bps.rollout(param_vec, rollout_rng)
    assert reward.shape == ()

    # Check evaluation
    rng, eval_rng = jax.random.split(rng)
    metrics = bps.evaluate(param_vec, eval_rng)
    assert isinstance(metrics, dict)
    assert "eval/episode_reward" in metrics

    # Check that the main training loop works
    params = bps.train(iterations=100, num_evals=3)
    assert params is not None


if __name__ == "__main__":
    test_bps()
