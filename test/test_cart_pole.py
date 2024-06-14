import jax

from playground.envs.cart_pole.cart_pole_env import CartPoleSwingupEnv


def test_cart_pole_env():
    """Test that the cart-pole environment can be created."""
    env = CartPoleSwingupEnv()

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    assert state.obs.shape == (5,)
    assert state.reward == 0.0
    assert state.done == 0.0
    assert state.info["step"] == 0

    ctrl = jax.numpy.zeros(1)
    state = env.step(state, ctrl)

    assert state.reward != 0.0
    assert state.done == 0.0
    assert state.info["step"] == 1


if __name__ == "__main__":
    test_cart_pole_env()
