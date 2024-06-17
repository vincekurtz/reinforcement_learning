import pickle
import sys

import jax

from playground.architectures import MLP
from playground.envs.pendulum.pendulum_env import PendulumSwingupEnv
from playground.predictive_sampling import (
    PredictiveSampling,
    PredictiveSamplingOptions,
)
from playground.simulation import run_interactive

"""
Use predictive sampling regression to train a pendulum swingup task.
"""


def train():
    """Train the policy and save it to a file."""
    env = PendulumSwingupEnv()
    options = PredictiveSamplingOptions(
        episode_length=500,
        planning_horizon=40,
        num_envs=32,
        num_samples=32,
        noise_std=0.5,
        learning_rate=1e-3,
        batch_size=100,
        epochs_per_iteration=100,
        iterations=1,
    )
    policy = MLP(
        layer_sizes=(16, 16, options.planning_horizon * env.action_size)
    )

    ps = PredictiveSampling(env, policy, options)
    params = ps.train()

    fname = "/tmp/pendulum_ps.pkl"
    print(f"Saving policy to {fname}...")
    with open(fname, "wb") as f:
        data = {"params": params, "network": policy}
        pickle.dump(data, f)


def test():
    """Test the policy with an interactive mujoco simulation."""
    env = PendulumSwingupEnv()

    # Load the trained policy
    with open("/tmp/pendulum_ps.pkl", "rb") as f:
        data = pickle.load(f)
    policy_net = data["network"]
    params = data["params"]

    # Create a policy function
    def policy_function(obs):
        flat_actions = policy_net.apply(params, obs)
        actions = flat_actions.reshape(-1, env.action_size)
        return actions[0]

    # Run the sim
    run_interactive(env, jax.jit(policy_function), fixed_camera_id=0)


if __name__ == "__main__":
    usage_msg = "Usage: python pendulum_psr.py [train|test]"

    if len(sys.argv) != 2:
        print(usage_msg)
        sys.exit(1)

    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        test()
    else:
        print(usage_msg)
        sys.exit(1)
