import pickle
import sys

import jax

from playground.architectures import MLP
from playground.boltzmann import (
    BoltzmannPolicySearch,
    BoltzmannPolicySearchOptions,
)
from playground.envs.half_cheetah.half_cheetah_env import HalfCheetahEnv
from playground.simulation import run_interactive

"""
Use Boltzmann Policy Search to train a half-cheetah running task.
"""


def train():
    """Train the policy and save it to a file."""
    env = HalfCheetahEnv()
    policy_net = MLP(layer_sizes=(16, 16, 6))
    options = BoltzmannPolicySearchOptions(
        episode_length=100,
        num_envs=512,
        temperature=1.0,
        sigma=0.1,
    )

    bps = BoltzmannPolicySearch(
        env=env,
        policy=policy_net,
        options=options,
        tensorboard_logdir="/tmp/rl_playground/half_cheetah_bps",
    )
    params = bps.train(iterations=2000, num_evals=10)

    fname = "/tmp/half_cheetah_bps.pkl"
    print(f"Saving policy to {fname}...")
    with open(fname, "wb") as f:
        data = {"params": params, "network": policy_net}
        pickle.dump(data, f)


def test():
    """Test the policy with an interactive mujoco simulation."""
    env = HalfCheetahEnv()

    # Load the trained policy
    with open("/tmp/half_cheetah_bps.pkl", "rb") as f:
        data = pickle.load(f)
    policy_net = data["network"]
    params = data["params"]

    # Create a policy function
    jit_policy = jax.jit(lambda obs: policy_net.apply(params, obs))

    # Run the sim
    run_interactive(env, jit_policy, fixed_camera_id=0)


if __name__ == "__main__":
    usage_message = "Usage: python half_cheetah_bps.py [train|test]"

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
