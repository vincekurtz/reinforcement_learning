from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from flax import struct

from playground import ROOT


@struct.dataclass
class HalfCheetahConfig:
    """Config dataclass for the half-cheetah running task."""

    model_path: Union[Path, str] = ROOT + "/envs/half_cheetah/half_cheetah.xml"

    # Cost function coefficients
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 0.1

    # Reset parameters
    reset_noise_scale: float = 0.1


class HalfCheetahEnv(PipelineEnv):
    """Environment for training a half-cheetah to run.

    States: x = (qpos, qvel), shape=(18,)
    Observations: All states except for horizontal position, shape=(17,)
    Actions: torques applied to the joints, shape=(6,)
    """

    def __init__(self, config: Optional[HalfCheetahConfig] = None) -> None:
        """Initialize the half-cheetah running environment."""
        if config is None:
            config = HalfCheetahConfig()
        self.config = config
        mj_model = mujoco.MjModel.from_xml_path(config.model_path)
        sys = mjcf.load_model(mj_model)

        super().__init__(sys, n_frames=5, backend="mjx")

    def reset(self, rng: jax.random.PRNGKey) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # Randomize the initial state
        low, hi = -self.config.reset_noise_scale, self.config.reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = hi * jax.random.normal(rng2, (self.sys.qd_size(),))
        data = self.pipeline_init(qpos, qvel)

        # Other state fields
        obs = self._compute_obs(data, {})
        reward, done, zero = jnp.zeros(3)
        metrics = {
            "x_position": zero,
            "x_velocity": zero,
            "reward_ctrl": zero,
            "reward_run": zero,
        }
        state_info = {"rng": rng, "step": 0}
        return State(data, obs, reward, done, metrics, state_info)

    def step(self, state: State, action: jax.Array) -> State:
        """Take a step in the environment."""
        # Simulate physics
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        # Compute reward
        x_vel = (data.x.pos[0, 0] - data0.x.pos[0, 0]) / self.dt
        forward_reward = self.config.forward_reward_weight * x_vel
        ctrl_cost = self.config.ctrl_cost_weight * jnp.sum(jnp.square(action))
        reward = forward_reward - ctrl_cost

        # Compute observation
        obs = self._compute_obs(data, state.info)

        state.metrics.update(
            x_position=data.x.pos[0, 0],
            x_velocity=x_vel,
            reward_ctrl=-ctrl_cost,
            reward_run=forward_reward,
        )
        state.info["step"] += 1
        return state.replace(pipeline_state=data, obs=obs, reward=reward)

    def _compute_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Compute the observation from the current state."""
        position = data.qpos[1:]  # Skip the horizontal position
        velocity = data.qvel

        return jnp.concatenate([position, velocity])

    @property
    def observation_size(self) -> int:
        """Size of the observation space."""
        return 17

    @property
    def action_size(self) -> int:
        """Size of the action space."""
        return 6
