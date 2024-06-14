from pathlib import Path
from typing import Optional, Union

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
from brax import actuator, base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from flax import struct

from playground import ROOT


@struct.dataclass
class HumanoidStandupConfig:
    """Config dataclass for the humanoid standup task."""

    model_path: Union[Path, str] = ROOT + "/envs/humanoid/humanoidstandup.xml"

    # Cost function coefficients
    up_reward_weight: float = 1.0  # z / dt
    ctrl_cost_weight: float = 0.1  # -|| u ||^2

    # Reset parameters
    reset_noise_scale: float = 0.01


class HumanoidStandupEnv(PipelineEnv):
    """Environment for getting a humanoid to stand up.

    States: x = (q, v), shape=(47,)
    Observations: (q except base x/y position, v, shape=(45,)
    Actions: torques applied to the joints, shape=(17,)
    """

    def __init__(self, config: Optional[HumanoidStandupConfig] = None) -> None:
        """Initialize the humanoid standup environment."""
        if config is None:
            config = HumanoidStandupConfig()
        self.config = config
        mj_model = mujoco.MjModel.from_xml_path(config.model_path)
        sys = mjcf.load_model(mj_model)

        super().__init__(sys, n_frames=5, backend="mjx")

    def reset(self, rng: jax.random.PRNGKey) -> State:
        """Reset the environment to a new initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self.config.reset_noise_scale, self.config.reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )

        data = self.pipeline_init(qpos, qvel)
        obs = self._compute_obs(data, jnp.zeros(self.sys.act_size()))
        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_linup": zero,
            "reward_quadctrl": zero,
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Compute the next state and rewards."""
        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        # Step the physics
        data = self.pipeline_step(state.pipeline_state, action)

        # Compute reward
        z_pos = data.x.pos[0, 2]  # z coordinate of torso
        up_reward = (z_pos - 0) / self.dt
        ctrl_cost = -jnp.sum(jnp.square(action))
        reward = (
            self.config.up_reward_weight * up_reward
            + self.config.ctrl_cost_weight * ctrl_cost
        )

        # Compute the observation
        obs = self._compute_obs(data, action)

        # Update the state
        state.metrics.update(reward_linup=up_reward, reward_quadctrl=ctrl_cost)
        return state.replace(pipeline_state=data, obs=obs, reward=reward)

    def _compute_obs(self, data: mjx.Data, action: jnp.ndarray) -> jnp.ndarray:
        """Compute the observation from the current state."""
        position = data.q[2:]  # exclude x and y positions in the world
        velocity = data.qd

        return jnp.concatenate([position, velocity])
