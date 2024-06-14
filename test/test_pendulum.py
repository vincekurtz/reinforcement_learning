import jax
import mujoco

from playground import ROOT
from playground.envs.pendulum.pendulum_env import PendulumSwingupEnv


def test_pendulum_mujoco_model():
    """Test that the pendulum mujoco model can be loaded."""
    model_file = ROOT + "/envs/pendulum/scene.xml"

    model = mujoco.MjModel.from_xml_path(model_file)
    data = mujoco.MjData(model)

    assert isinstance(model, mujoco.MjModel)
    assert isinstance(data, mujoco.MjData)
    assert model.nq == 1
    assert model.nv == 1
    assert model.nu == 1
    assert data.qpos.shape == (1,)
    assert data.qvel.shape == (1,)
    assert data.ctrl.shape == (1,)


def test_pendulum_env():
    """Test that the pendulum environment can be created."""
    env = PendulumSwingupEnv()

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    assert state.obs.shape == (3,)
    assert state.reward == 0.0
    assert state.done == 0.0
    assert state.info["step"] == 0

    ctrl = jax.numpy.zeros(1)
    state = env.step(state, ctrl)

    assert state.reward != 0.0
    assert state.done == 0.0
    assert state.info["step"] == 1


if __name__ == "__main__":
    test_pendulum_mujoco_model()
    test_pendulum_env()
