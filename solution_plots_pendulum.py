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

@torch.no_grad
def plot_switching_coefficient(model, history_length=1, surface_index=0):
    """
    Make a contour plot of the switching surface. Just uses the first
    observation, and pads the rest with zeros.
    """
    def compute_switching_coefficient(obs):
        """
        A little helper function to compute the value for a given observation.
        Handles conversion between torch and numpy.
        """
        obs = torch.from_numpy(obs).float().to(model.device)
        sigma = model.policy.mlp_extractor.chooser(obs)
        return sigma.cpu().numpy()

    # Sample a bunch of initial states
    n = 150
    thetas = np.linspace(-np.pi, 2*np.pi, n)
    theta_dots = np.linspace(-8, 8, n)

    # Set up a grid of observations
    theta_grid, theta_dot_grid = np.meshgrid(thetas, theta_dots)
    obs = np.zeros((n*n, 3*history_length))
    obs[:,0] = np.cos(theta_grid.flatten())
    obs[:,1] = np.sin(theta_grid.flatten())
    obs[:,2] = theta_dot_grid.flatten()

    # Compute the switching coefficient for each observation
    sigmas = compute_switching_coefficient(obs)[:,surface_index]
    sigma_grid = sigmas.reshape((n,n))

    # Plot the switching surface
    plt.contourf(theta_grid, theta_dot_grid, sigma_grid, 20, cmap='binary', 
                 alpha=0.7)
    plt.colorbar(label="Switching coefficient")

@torch.no_grad
def plot_value_function(model, history_length=1):
    """
    Make a contour plot of the value function.
    """
    def compute_value(obs):
        """
        A little helper function to compute the value for a given observation.
        Handles conversion between torch and numpy.
        """
        obs = torch.from_numpy(obs).float().to(model.device)
        value_hidden = model.policy.mlp_extractor.forward_critic(obs)
        value = model.policy.value_net(value_hidden)
        return value.cpu().numpy()
    
    # Sample a bunch of initial states
    n = 150
    thetas = np.linspace(-np.pi, 2*np.pi, n)
    theta_dots = np.linspace(-8, 8, n)

    # Set up a grid of observations
    theta_grid, theta_dot_grid = np.meshgrid(thetas, theta_dots)
    obs = np.zeros((n*n, 3 * history_length))
    obs[:,0] = np.cos(theta_grid.flatten())
    obs[:,1] = np.sin(theta_grid.flatten())
    obs[:,2] = theta_dot_grid.flatten()

    # Compute the value for each observation
    values = compute_value(obs)
    value_grid = values.reshape((n,n))

    # Plot the value function
    plt.contourf(theta_grid, theta_dot_grid, value_grid, 20, cmap='RdYlBu')
    plt.colorbar(label="Value")

if __name__=="__main__":
    model = PPO.load("trained_models/pendulum")

    plt.figure()
    plot_value_function(model)
    plot_pendulum_vector_field()

    #plt.figure()
    #plot_switching_coefficient(model, surface_index=2)
    #plot_pendulum_vector_field()

    plt.show()
