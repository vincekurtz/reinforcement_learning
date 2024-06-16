import jax

from playground.architectures import MLP
from playground.envs.pendulum.pendulum_env import PendulumSwingupEnv
from playground.predictive_sampling import (
    PredictiveSampling,
    PredictiveSamplingOptions,
)


def test_predictive_sampling():
    """Test predictive sampling policy search."""
    env = PendulumSwingupEnv()
    options = PredictiveSamplingOptions(
        episode_length=100, planning_horizon=10, num_envs=4, num_samples=8
    )
    policy = MLP(layer_sizes=(8, 8, options.planning_horizon * env.action_size))

    ps = PredictiveSampling(env, policy, options)
    print(ps)


if __name__ == "__main__":
    test_predictive_sampling()
