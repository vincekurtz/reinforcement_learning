#!/usr/bin/env python

##
#
# Plot one-step prediction errors for the pendulum example.
#
# Assumes data has been generated by running pendulum.py with --test.
#
##

import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_errors(file, label):
    """Plot the prediction errors from the given file."""
    with open(file, "rb") as f:
        prediction_errors = pickle.load(f)

    # Plot the prediction errors
    plt.plot(prediction_errors, label=label, linewidth=2)

if __name__=="__main__":
    plot_prediction_errors("data/prediction_errors.pkl", label="Normal Gravity")
    plot_prediction_errors("data/prediction_errors_mars.pkl", label="Mars Gravity")
    plt.xlabel("Time step")
    plt.ylabel("One-Step Predition Error")
    plt.legend()
    plt.show()