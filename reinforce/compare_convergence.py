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

def make_a_plot(data):
    # Set up the plot
    plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("Reward")

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
    plt.legend()

if __name__=="__main__":
    # Dictionary containing the name of each run and the location of the pickle
    depth_data = {"2x1 (19 params)": "2x1_plotter.pkl",
            "2x2 (49 params)": "2x2_plotter.pkl",
            "2x3 (81 params)": "2x3_plotter.pkl",
            "2x4 (113 params)": "2x4_plotter.pkl",
            "2x5 (145 params)": "2x5_plotter.pkl"}

    bredth_data = {"2x1 (19 params)": "2x1_plotter.pkl",
            "3x1 (29 params)": "3x1_plotter.pkl",
            "4x1 (41 params)": "4x1_plotter.pkl",
            "5x1 (55 params)": "5x1_plotter.pkl",
            "6x1 (71 params)": "6x1_plotter.pkl",
            "7x1 (89 params)": "7x1_plotter.pkl",
            "8x1 (109 params)": "8x1_plotter.pkl"}

    comparison_data = {
            "2x4 (113 params)": "2x4_plotter.pkl",
            "8x1 (109 params)": "8x1_plotter.pkl"}


    make_a_plot(depth_data)
    make_a_plot(bredth_data)
    make_a_plot(comparison_data)

    plt.show() 


