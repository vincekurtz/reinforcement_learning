#!/usr/bin/env python

##
#
# Try to analyze the learned lifted dynamics of the pendulum, to see whether
# we've learned a Koopman representation for the closed-loop system.
#
##

import gymnasium as gym
from stable_baselines3 import PPO

from solution_plots_pendulum import plot_pendulum_vector_field

import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the learned model
model = PPO.load("trained_models/pendulum")
phi = model.policy.mlp_extractor.lifting_function

# Create the environment
env = gym.make("Pendulum-v1")

# Gather some data by running the policy
num_traj = 10
steps_per_traj = 100
X = np.zeros((num_traj, steps_per_traj, 3))
    
for traj in range(num_traj):
    # Reset to a new initial state
    obs, _ = env.reset()

    for step in range(steps_per_traj):
        # Store the current observation [cos(theta), sin(theta), theta_dot]
        X[traj, step, :] = obs

        # Step the environment
        action, _ = model.predict(obs)
        obs, _, _, _, _ = env.step(action)

# Compute phi(x) for each observation in the dataset
with torch.no_grad():
    X_torch = torch.from_numpy(X).float().to(model.device)
    phi_X = phi(X_torch).cpu().numpy()

# Compute a linear least-squares fit for the lifted dynamics, 
Z = phi_X.reshape((num_traj*steps_per_traj, -1))
Z_now = Z[0:-1,:]
Z_next = Z[1:,:]

A, residuals, rank, s = np.linalg.lstsq(Z_now, Z_next)

# For comparison, plot the lifted state for a single trajectory
plt.figure()
z = phi_X[0,:,:]  # The actual lifted state
z_pred = np.zeros_like(z)  # The predicted lifted state
z_pred[0,:] = z[0,:]  # Set the initial condition
for step in range(1,steps_per_traj):
    z_pred[step,:] = A @ z_pred[step-1,:]

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.plot(z[:,i], label="actual")
    plt.plot(z_pred[:,i], label="predicted")

plt.legend()
plt.show()

# Plot observations over time
#plt.figure()
#
#plt.subplot(3,1,1)
#for traj in range(num_traj):
#    plt.plot(X[traj,:,0])
#plt.ylabel("cos(theta)")
#
#plt.subplot(3,1,2)
#for traj in range(num_traj):
#    plt.plot(X[traj,:,1])
#plt.ylabel("sin(theta)")
#
#plt.subplot(3,1,3)
#for traj in range(num_traj):
#    plt.plot(X[traj,:,2])
#plt.ylabel("theta_dot")
#
#plt.xlabel("Timestep")
#plt.show()
