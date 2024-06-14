import pickle
import sys

import jax

from playground.architectures import MLP
from playground.boltzmann import (
    BoltzmannPolicySearch,
    BoltzmannPolicySearchOptions,
)
from playground.envs.cart_pole.cart_pole_env import CartPoleSwingupEnv
from playground.simulation import run_interactive

"""
Use Boltzmann Policy Search to train a cart-pole swingup task.
"""


def train():
    """Train the policy and save it to a file."""
    env = CartPoleSwingupEnv()
    policy_net = MLP(layer_sizes=(8, 8, 1))
    options = BoltzmannPolicySearchOptions(
        episode_length=100,
        num_envs=512,
        temperature=1.0,
        sigma=0.1,
    )

    bps = BoltzmannPolicySearch(env=env, policy=policy_net, options=options)
    params = bps.train(iterations=2000, num_evals=10)

    fname = "/tmp/cart_pole_bps.pkl"
    print(f"Saving policy to {fname}...")
    with open(fname, "wb") as f:
        data = {"params": params, "network": policy_net}
        pickle.dump(data, f)


def test():
    """Test the policy with an interactive mujoco simulation."""
    env = CartPoleSwingupEnv()

    # Load the trained policy
    with open("/tmp/cart_pole_bps.pkl", "rb") as f:
        data = pickle.load(f)
    policy_net = data["network"]
    params = data["params"]

    # Create a policy function
    jit_policy = jax.jit(lambda obs: policy_net.apply(params, obs))

    # Run the sim
    run_interactive(env, jit_policy, fixed_camera_id=0)


if __name__ == "__main__":
    usage_message = "Usage: python cart_pole_bps.py [train|test]"

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
