import time
from typing import Callable

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from brax.envs.base import PipelineEnv
from mujoco import mjx


def run_interactive(
    env: PipelineEnv,
    policy: Callable[[jnp.ndarray], jnp.ndarray],
    fixed_camera_id: int = None,
):
    """Run an interactive mujoco simulation with a given policy.

    Args:
        env: The environment to simulate.
        policy: A function that takes an observation and returns an action.
        fixed_camera_id: The camera ID to use, or None for the default camera.
    """
    # Extract a mujoco system model
    mj_model = env.sys.mj_model
    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)

    # Make an observation function
    # N.B. This takes an MJX data object and a dictionary of extra info.
    jit_obs = jax.jit(env._compute_obs)

    # Start the interactive simulation
    dt = float(env.dt)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        if fixed_camera_id is not None:
            # Set the custom camera
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        while viewer.is_running():
            start_time = time.time()

            # Get an observation from the environment
            mjx_data = mjx_data.replace(qpos=mj_data.qpos, qvel=mj_data.qvel)
            obs = jit_obs(mjx_data, {})

            # Get an action from the policy
            act = policy(obs)

            # Apply the action and step the simulation
            mj_data.ctrl[:] = np.array(act)
            for _ in range(env._n_frames):
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
