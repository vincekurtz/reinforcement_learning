#!/usr/bin/env python

##
#
# Make some plots to try to understand what the solution looks like for the
# simple pendulum example.
#
##

import gymnasium as gym
from stable_baselines3 import PPO
    
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_pendulum_vector_field(n=25):
    env = gym.make("Pendulum-v1")
    env.reset()

    thetas = np.linspace(-np.pi, 2*np.pi, n)
    theta_dots = np.linspace(-8, 8, n)

    for theta in thetas:
        for theta_dot in theta_dots:
            env.unwrapped.state = np.array([theta, theta_dot])
            env.step([0])
            dtheta = (env.unwrapped.state[0] - theta)
            dtheta_dot = (env.unwrapped.state[1] - theta_dot)

            plt.arrow(theta, theta_dot, dtheta, dtheta_dot,
                    head_width=0.05, head_length=0.1, color='blue', alpha=0.5)

    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")

def plot_learned_switching_surface(model):
    switching_model = model.policy.mlp_extractor.policy_net.chooser

    # Sample a bunch of initial states
    n = 50
    thetas = np.linspace(-np.pi, 2*np.pi, n)
    theta_dots = np.linspace(-6, 6, n)

    # Simulate the system for a while to get a history of observations
    for theta_init in thetas:
        for theta_dot_init in theta_dots:
            num_steps = 1
            history = np.zeros((10, 3))
            env = gym.make("Pendulum-v1")
            env.reset()
            env.unwrapped.state = np.array([theta_init, theta_dot_init])

            for t in range(num_steps):
                obs, _, _, _, _ = env.step([0])
                history = np.roll(history, shift=1, axis=0)
                history[0] = obs
            final_state = env.unwrapped.state

            with torch.no_grad():
                observation = history.flatten()
                observation = torch.from_numpy(observation).float().to(model.device)
                sigma = switching_model(observation).cpu().numpy()

            # Plot a dot shaded by the value of sigma
            #plt.scatter([theta_init], [theta_dot_init], color='r')
            plt.scatter(final_state[0], final_state[1], color='blue', edgecolors=None, alpha=sigma[1]/2)

@torch.no_grad
def plot_value_function(model):
    """
    Make a contour plot of the value function. Assumes that the model was
    trained with history_length=1 (i.e., no history).
    """
    def compute_value(obs):
        """
        A little helper function to compute the value for a given observation.
        Handles conversion between torch and numpy.
        """
        obs = torch.from_numpy(obs).float().to(model.device)
        value_hidden = model.policy.mlp_extractor.value_net(obs)
        value = model.policy.value_net(value_hidden)
        return value.cpu().numpy()
    
    # Sample a bunch of initial states
    n = 150
    thetas = np.linspace(-np.pi, 2*np.pi, n)
    theta_dots = np.linspace(-8, 8, n)

    # Set up a grid of observations
    theta_grid, theta_dot_grid = np.meshgrid(thetas, theta_dots)
    obs = np.zeros((n*n, 3))
    obs[:,0] = np.cos(theta_grid.flatten())
    obs[:,1] = np.sin(theta_grid.flatten())
    obs[:,2] = theta_dot_grid.flatten()

    # Compute the value for each observation
    values = compute_value(obs)
    value_grid = values.reshape((n,n))

    # Plot the value function
    plt.contourf(theta_grid, theta_dot_grid, value_grid, 20, cmap='RdYlBu')

    # Add a little scale bar with label
    plt.colorbar(label="Value")



if __name__=="__main__":
    model = PPO.load("trained_models/pendulum")
    plot_value_function(model)
    plot_pendulum_vector_field()

    # Plot the switching surface for the learned policy
    #plot_learned_switching_surface(model)


    plt.show()
