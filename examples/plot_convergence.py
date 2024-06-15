import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

"""
Tools for plotting convergence from a tensorboard event file.
"""


def plot_convergence(fname, fill_std=True, label=None):
    """Plot the reward and std from a TOPS tensorboard event file."""
    # Load the tensorboard event file
    ea = event_accumulator.EventAccumulator(fname)
    ea.Reload()

    # Get the quantities of interest as numpy arrays
    reward = [scalar.value for scalar in ea.Scalars("eval/episode_reward")]
    reward_std = [
        scalar.value for scalar in ea.Scalars("eval/episode_reward_std")
    ]
    steps = [scalar.step for scalar in ea.Scalars("eval/episode_reward")]

    reward = np.array(reward)
    reward_std = np.array(reward_std)

    # Plot the reward and std
    plt.plot(steps, reward, label=label, linewidth=3)
    if fill_std:
        plt.fill_between(
            steps, reward - reward_std, reward + reward_std, alpha=0.2
        )


if __name__ == "__main__":
    pendulum_ppo_log = "/tmp/rl_playground/pendulum_ppo/events.out.tfevents.1718486394.XPS-8960"
    pendulum_bpg_log = "/tmp/rl_playground/pendulum_bpg/events.out.tfevents.1718486394.XPS-8960"

    plt.figure(figsize=(10, 5))
    plot_convergence(pendulum_ppo_log, label="PPO")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()
