#!/usr/bin/env python

##
#
# Quick script for making convergence plots based on saved tensorboard data.
#
##

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_mean_reward_per_step(log_dir):
    """
    Load the average reward at ech step from the given tensorboard log.

    Args:
        log_dir (str): Path to the tensorboard log directory

    Returns:
        mean_reward (np.array): The mean reward at each timestep
        steps (np.array): The timestep at which each mean reward was computed
    """
    ea = EventAccumulator(log_dir)
    ea.Reload()

    # Get the average reward at each timestep
    rew_mean_data = np.array(ea.Scalars("rollout/ep_rew_mean"))
    reward = []
    steps = []
    for i in range(len(rew_mean_data)):
        reward.append(rew_mean_data[i].value)
        steps.append(rew_mean_data[i].step)

    return np.array(reward), np.array(steps)

def plot_learning_curve(log_dirs, label):
    """
    Make a plot of the learning curve for the given settings, averaged over
    several random seeds. 

    Args:
        log_dirs (list of str): Paths to the tensorboard log directories for 
                                runs with different random seeds
        label (str): Label for the curve 
    """
    reward = []
    steps = []
    for log_dir in log_dirs:
        rew, step = get_mean_reward_per_step(log_dir)
        reward.append(rew)
        steps.append(step)
    reward = np.array(reward)
    steps = np.array(steps)

    # Check that the steps are the same for all runs
    assert np.all([np.all(steps[i,:] == steps[0,:]) for i in range(len(log_dirs))])

    # Average over the runs
    mean_reward = np.mean(reward, axis=0)
    std_reward = np.std(reward, axis=0)

    # Plot the mean and standard deviation
    plt.plot(steps[0][:], mean_reward, label=label)
    plt.fill_between(steps[0][:], mean_reward - std_reward, 
                     mean_reward + std_reward, alpha=0.3)


if __name__=="__main__":
    koopman_log_dirs = ["/tmp/pendulum_tensorboard/PPO_1",
                "/tmp/pendulum_tensorboard/PPO_2",
                "/tmp/pendulum_tensorboard/PPO_3"]
    baseline_log_dirs = ["/tmp/pendulum_tensorboard/PPO_4",
                         "/tmp/pendulum_tensorboard/PPO_5",
                         "/tmp/pendulum_tensorboard/PPO_6"]
    plot_learning_curve(koopman_log_dirs, "Koopman RL")
    plot_learning_curve(baseline_log_dirs, "Standard RL")

    # Use exponential notation on x axis
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    plt.title("Pendulum Learning Curves, 3 random seeds")
    plt.ylabel("Average Reward")
    plt.xlabel("Total Simulation Timesteps")
    plt.legend()
    plt.show()
