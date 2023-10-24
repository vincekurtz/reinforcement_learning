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
                sigma = switching_model(observation).item()

            # Plot a dot shaded by the value of sigma
            #plt.scatter([theta_init], [theta_dot_init], color='r')
            plt.scatter(final_state[0], final_state[1], color='blue', edgecolors=None, alpha=sigma/2)

if __name__=="__main__":
    # Plot the vector field for the pendulum
    plot_pendulum_vector_field()

    # Plot the switching surface for the learned policy
    model = PPO.load("pendulum")
    plot_learned_switching_surface(model)

    plt.show()
