import time
from typing import Callable

import jax
import jax.numpy as jnp
import mediapy
import mujoco
import mujoco.viewer
import numpy as np
from brax.envs.base import PipelineEnv
from mujoco import mjx


def save_video(
    env: PipelineEnv,
    policy: Callable[[jnp.ndarray], jnp.ndarray],
    filename: str,
    duration: float = 10.0,
    camera_name: str = None,
    seed: int = 0,
):
    """Make a video of an MJX simulation with a given policy.

    Args:
        env: The environment to simulate.
        policy: A function that takes an observation and returns an action.
        filename: The filename to save the video to.
        duration: The duration of the video in seconds.
        camera_name: The name of the camera to use, or None for the default.
        seed: The random seed to use.
    """
    rng = jax.random.PRNGKey(seed)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    print("Simulating...")
    num_steps = int(duration / env.dt)
    for _ in range(num_steps):
        action = policy(state.obs)
        state = jit_step(state, action)
        rollout.append(state.pipeline_state)

    print("Rendering frames...")
    frames = env.render(rollout, camera=camera_name)

    print(f"Writing video to {filename}...")
    mediapy.write_video(filename, frames, fps=1 / env.dt)


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
