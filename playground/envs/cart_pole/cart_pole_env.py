from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from flax import struct
from mujoco import mjx

from playground import ROOT


@struct.dataclass
class CartPoleSwingupConfig:
    """Config dataclass for cart-pole swingup."""

    # model path
    model_path: Union[Path, str] = ROOT + "/envs/cart_pole/scene.xml"

    # number of "simulation steps" for every control input
    physics_steps_per_control_step: int = 1

    # Reward function coefficients
    upright_angle_cost: float = 1.0
    center_cart_cost: float = 0.1
    velocity_cost: float = 0.01
    control_cost: float = 0.001

    # Ranges for sampling initial conditions
    pos_hi: float = 1.0
    pos_lo: float = -1.0
    theta_hi: float = jnp.pi
    theta_lo: float = -jnp.pi
    qvel_hi: float = 10.0
    qvel_lo: float = -10.0


class CartPoleSwingupEnv(PipelineEnv):
    """Environment for training a cart-pole swingup task.

    States: x = (pos, theta, vel, dtheta), shape=(4,)
    Observations: y = (pos, cos(theta), sin(theta), vel, dtheta), shape=(5,)
    Actions: a = tau, the force on the cart, shape=(1,)
    """

    def __init__(self, config: Optional[CartPoleSwingupConfig] = None) -> None:
        """Initialize the swingup env."""
        if config is None:
            config = CartPoleSwingupConfig()
        self.config = config
        mj_model = mujoco.MjModel.from_xml_path(config.model_path)
        sys = mjcf.load_model(mj_model)

        super().__init__(
            sys, n_frames=config.physics_steps_per_control_step, backend="mjx"
        )

    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to a new initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # Reset positions and velocities
        qpos_lo = jnp.array([self.config.pos_lo, self.config.theta_lo])
        qpos_hi = jnp.array([self.config.pos_hi, self.config.theta_hi])
        qpos = jax.random.uniform(
            rng1, (self.sys.nq,), minval=qpos_lo, maxval=qpos_hi
        )
        qvel = jax.random.uniform(
            rng2,
            (self.sys.nv,),
            minval=self.config.qvel_lo,
            maxval=self.config.qvel_hi,
        )
        data = self.pipeline_init(qpos, qvel)

        # Other state fields
        obs = self._compute_obs(data, {})
        reward, done = jnp.zeros(2)
        metrics = {
            "upright_reward": 0.0,
            "center_cart_reward": 0.0,
            "velocity_reward": 0.0,
            "control_reward": 0.0,
        }
        state_info = {"rng": rng, "step": 0}
        return State(data, obs, reward, done, metrics, state_info)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Steps the environment forward one timestep."""
        data = self.pipeline_step(state.pipeline_state, action)
        obs = self._compute_obs(data, state.info)

        # Compute a normalized angle error (upright is zero)
        pos = data.qpos[0]
        theta = data.qpos[1]
        theta_err = jnp.array([jnp.cos(theta) - 1, jnp.sin(theta)])

        # Compute the reward
        upright_reward = -jnp.square(theta_err).sum()
        center_cart_reward = -jnp.square(jnp.square(pos)).sum()
        velocity_reward = -jnp.square(data.qvel).sum()
        control_reward = -jnp.square(data.ctrl).sum()

        reward = (
            self.config.upright_angle_cost * upright_reward
            + self.config.center_cart_cost * center_cart_reward
            + self.config.velocity_cost * velocity_reward
            + self.config.control_cost * control_reward
        )

        # Update the metrics and extra info
        state.info["step"] += 1
        state.metrics["upright_reward"] = upright_reward
        state.metrics["center_cart_reward"] = center_cart_reward
        state.metrics["velocity_reward"] = velocity_reward
        state.metrics["control_reward"] = control_reward
        return state.replace(pipeline_state=data, obs=obs, reward=reward)

    def _compute_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Computes the observation from the state."""
        p = data.qpos[0]
        s = jnp.sin(data.qpos[1])
        c = jnp.cos(data.qpos[1])
        pd = data.qvel[0]
        td = data.qvel[1]
        return jnp.array([p, c, s, pd, td])

    @property
    def action_size(self) -> int:
        """Returns the size of the action space."""
        return 1

    @property
    def observation_size(self) -> int:
        """Returns the size of the observation space."""
        return 5
