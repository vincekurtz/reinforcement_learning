import pickle
import sys
import time

import jax
import mediapy
import mujoco
import mujoco.viewer
import numpy as np
from brax import envs
from brax.training.distribution import NormalTanhDistribution
from mujoco import mjx

from playground.architectures import MLP
from playground.envs.pendulum.pendulum_env import PendulumSwingupEnv
from playground.ppo import (
    BraxPPONetworksWrapper,
    make_policy_function,
    train_ppo,
)

"""
Use standard PPO to train a pendulum swingup task.
"""


def train(
    num_timesteps=20_000_000,
    num_evals=10,
    reward_scaling=1.0,
    episode_length=100,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=32,
    num_updates_per_batch=8,
    discounting=0.97,
    learning_rate=1e-3,
    clipping_epsilon=0.2,
    entropy_cost=1e-3,
    num_envs=2048,
    batch_size=1024,
    seed=0,
):
    """Train the swingup policy and save it to a file."""
    # Create policy and value functions
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=MLP(layer_sizes=(32, 32, 2)),
        value_network=MLP(layer_sizes=(64, 64, 1)),
        action_distribution=NormalTanhDistribution,
    )

    # Train the policy
    train_ppo(
        env=PendulumSwingupEnv,
        network_wrapper=network_wrapper,
        save_path="/tmp/pendulum_ppo.pkl",
        num_timesteps=num_timesteps,
        num_evals=num_evals,
        reward_scaling=reward_scaling,
        episode_length=episode_length,
        normalize_observations=normalize_observations,
        action_repeat=action_repeat,
        unroll_length=unroll_length,
        num_minibatches=num_minibatches,
        num_updates_per_batch=num_updates_per_batch,
        discounting=discounting,
        learning_rate=learning_rate,
        clipping_epsilon=clipping_epsilon,
        entropy_cost=entropy_cost,
        num_envs=num_envs,
        batch_size=batch_size,
        seed=seed,
    )


def test(record=False, record_path="pendulum_ppo.mp4", record_seconds=10):
    """Test the swingup policy with an interactive mujoco simulation."""
    # Create a brax environment
    envs.register_environment("pendulum_swingup", PendulumSwingupEnv)
    env = envs.get_environment("pendulum_swingup")

    # Extract the mujoco system model
    mj_model = env.sys.mj_model
    mj_data = mujoco.MjData(mj_model)
    mj_data.qvel[:] = np.array([-2.0])

    # Set up a renderer for saving a video
    renderer = mujoco.Renderer(mj_model, 1080, 1920)

    # Load the trained policy
    with open("/tmp/pendulum_ppo.pkl", "rb") as f:
        network_and_params = pickle.load(f)
    network_wrapper = network_and_params["network_wrapper"]
    params = network_and_params["params"]

    # Create a policy function
    policy = make_policy_function(
        network_wrapper=network_wrapper,
        params=params,
        observation_size=3,
        action_size=1,
        normalize_observations=True,
    )
    jit_policy = jax.jit(policy)

    # Start an interactive simulation
    rng = jax.random.PRNGKey(0)
    dt = float(env.dt)
    render_frames = []
    sim_seconds = 0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Use the fixed camera
        viewer.cam.fixedcamid = 0
        viewer.cam.type = 2

        while viewer.is_running():
            start_time = time.time()
            act_rng, rng = jax.random.split(rng)

            # Get an observation from the environment
            obs = env._compute_obs(mjx.put_data(mj_model, mj_data), {})

            # Get an action from the policy
            act, _ = jit_policy(obs, act_rng)
            act = env.config.u_max * act

            # Apply the policy and step the simulation
            mj_data.ctrl[:] = np.array(act)
            for _ in range(env._n_frames):
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

            if record:
                # Render the simulation
                renderer.update_scene(mj_data, camera=0)
                pixels = renderer.render()
                render_frames.append(pixels)

                # Stop recording after a few seconds
                if sim_seconds > record_seconds:
                    viewer.close()
                    break

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
            sim_seconds += dt

    # Save the video
    if record:
        print(f"Saving video to {record_path}...")
        mediapy.write_video(record_path, render_frames, fps=50)


if __name__ == "__main__":
    usage_message = "Usage: python pendulum_ppo.py [train|test]"

    if len(sys.argv) != 2:
        print(usage_message)
        sys.exit(1)

    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        test()
    else:
        print(usage_message)
        sys.exit(1)
