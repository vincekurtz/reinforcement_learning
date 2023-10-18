#!/usr/bin/env python

##
#
# A quick script to compare training convergence for a few different runs.
# Assumes that convergence data has been saved as a pickled ConvergencePlotter
# object.
#
##

import matplotlib.pyplot as plt
import numpy as np
import pickle

# Dictionary containing the name of each run and the location of the pickle
data = {"4 lifted states": "koopman4/plotter.pkl",
        "8 lifted states": "koopman8/plotter.pkl",
        "10 lifted states": "koopman10/plotter.pkl",
        "12 lifted states": "koopman12/plotter.pkl",
        "16 lifted states": "koopman16/plotter.pkl",
        "32 lifted states": "koopman32/plotter.pkl"}

# Set up the plot
plt.figure()
plt.xlabel("Iteration")
plt.ylabel("Reward")

state_size = [4, 8, 10, 12, 16, 32]
final_reward = []

for name, fname in data.items():
    with open(fname, "rb") as f:
        plotter = pickle.load(f)

    mean_rewards = [np.mean(rewards) for rewards in plotter.rewards]
    std_rewards = [np.std(rewards) for rewards in plotter.rewards]

    # Make a plot of the mean, with a shaded region for the standard deviation
    plt.plot(plotter.iteration_numbers, mean_rewards, label=name)
    plt.fill_between(plotter.iteration_numbers,
                     np.array(mean_rewards) - np.array(std_rewards),
                     np.array(mean_rewards) + np.array(std_rewards),
                     alpha=0.2)

    final_reward.append(mean_rewards[-1])

plt.legend()

# Make a second figure comparing the final reward only
plt.figure()
plt.plot(state_size, final_reward, "o")
plt.xlabel("Size of the lifted state")
plt.ylabel("Final reward")

plt.show() 


