import jax

from playground.envs.half_cheetah.half_cheetah_env import HalfCheetahEnv


def test_half_cheetah_env():
    """Test that the environment can be created."""
    env = HalfCheetahEnv()

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    assert state.obs.shape == (17,)
    assert state.reward == 0.0
    assert state.done == 0.0
    assert state.info["step"] == 0

    ctrl = jax.numpy.zeros(6)
    state = env.step(state, ctrl)

    assert state.reward != 0.0
    assert state.done == 0.0
    assert state.info["step"] == 1


if __name__ == "__main__":
    test_half_cheetah_env()
