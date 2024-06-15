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
    start_time = ea.Scalars("eval/episode_reward")[0].wall_time
    times = [
        scalar.wall_time - start_time
        for scalar in ea.Scalars("eval/episode_reward")
    ]

    reward = np.array(reward)
    reward_std = np.array(reward_std)
    steps = np.array(steps)
    times = np.array(times)

    # Plot the reward and std
    plt.plot(times, reward, label=label, linewidth=3)
    if fill_std:
        plt.fill_between(
            times, reward - reward_std, reward + reward_std, alpha=0.2
        )


if __name__ == "__main__":
    pendulum_ppo_log = "/tmp/rl_playground/pendulum_ppo/events.out.tfevents.1718486394.XPS-8960"
    pendulum_bps_log = "/tmp/rl_playground/pendulum_bps/events.out.tfevents.1718486434.XPS-8960"

    cart_pole_ppo_log = "/tmp/rl_playground/cart_pole_ppo/events.out.tfevents.1718486321.XPS-8960"
    cart_pole_bps_log = "/tmp/rl_playground/cart_pole_bps/events.out.tfevents.1718486356.XPS-8960"

    half_cheetah_ppo_log = "/tmp/rl_playground/half_cheetah_ppo/events.out.tfevents.1718486159.XPS-8960"
    half_cheetah_bps_log = "/tmp/rl_playground/half_cheetah_bps/events.out.tfevents.1718485985.XPS-8960"

    humanoid_ppo_log = "/tmp/rl_playground/humanoid_ppo/events.out.tfevents.1718484961.XPS-8960"
    humanoid_bps_log = "/tmp/rl_playground/humanoid_bps/events.out.tfevents.1718489357.XPS-8960"

    plt.figure(figsize=(10, 5))

    plt.subplot(2, 2, 1)
    plt.title("Pendulum")
    plot_convergence(pendulum_ppo_log, label="PPO")
    plot_convergence(pendulum_bps_log, label="BPS")
    plt.xlabel("Wall Clock Time (s)")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("Cart Pole")
    plot_convergence(cart_pole_ppo_log, label="PPO")
    plot_convergence(cart_pole_bps_log, label="BPS")
    plt.xlabel("Wall Clock Time (s)")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("Half Cheetah")
    plot_convergence(half_cheetah_ppo_log, label="PPO")
    plot_convergence(half_cheetah_bps_log, label="BPS")
    plt.xlabel("Wall Clock Time (s)")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title("Humanoid")
    plot_convergence(humanoid_ppo_log, label="PPO")
    plot_convergence(humanoid_bps_log, label="BPS")
    plt.xlabel("Wall Clock Time (s)")
    plt.ylabel("Reward")
    plt.legend()

    plt.show()
