from pathlib import Path

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

    # Set up a temporary tensorboard logdir
    local_dir = Path("_test_bps")
    local_dir.mkdir(parents=True, exist_ok=True)

    options = BoltzmannPolicySearchOptions(
        episode_length=100,
        num_envs=32,
        temperature=1.0,
        sigma=0.1,
    )
    bps = BoltzmannPolicySearch(
        env, policy, options, tensorboard_logdir=local_dir
    )
    assert bps.num_params > 0

    # Check that a single rollout works
    rng, rollout_rng, init_rng = jax.random.split(rng, 3)
    train_state = bps.make_training_state(init_rng)
    param_vec = train_state.param_vector
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

    # Clean up
    for p in local_dir.iterdir():
        p.unlink()
    local_dir.rmdir()


if __name__ == "__main__":
    test_bps()
