import jax

from playground.envs.humanoid.standup import HumanoidStandupEnv


def test_humanoid_standup_env():
    """Test that the environment can be created."""
    env = HumanoidStandupEnv()

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    assert state.obs.shape == (45,)
    assert state.reward == 0.0
    assert state.done == 0.0
    assert "reward_linup" in state.metrics
    assert "reward_quadctrl" in state.metrics

    ctrl = jax.numpy.ones(17)
    state = env.step(state, ctrl)

    assert state.reward != 0.0
    assert state.done == 0.0


if __name__ == "__main__":
    test_humanoid_standup_env()
