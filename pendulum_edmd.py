#!/usr/bin/env python

##
#
# Try to analyze the learned lifted dynamics of the pendulum, to see whether
# we've learned a Koopman representation for the closed-loop system.
#
##

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from solution_plots_pendulum import plot_pendulum_vector_field

import torch
import numpy as np
import matplotlib.pyplot as plt

SEED = 0
np.random.seed(SEED)
set_random_seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Load the learned model
model = PPO.load("trained_models/pendulum")
phi = model.policy.mlp_extractor.lifting_function

# Create the environment
env = gym.make("Pendulum-v1")
env.action_space.seed(SEED)

# Gather some data by running the policy
print("Gathering Data")
num_traj = 100
steps_per_traj = 200
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
#  z_t = z_{t+1} A  (awkward order to fit with numpy least squares)
# where z_t = phi(x_t).
print("Computing Linear Least-Squares Fit")
Z = phi_X.reshape((num_traj*steps_per_traj, -1))
Z_now = Z[0:-1,:]
Z_next = Z[1:,:]
A, residuals, rank, s = np.linalg.lstsq(Z_now, Z_next, rcond=1)
print("residual: ", np.linalg.norm(residuals))

# Compute a linear least-squares fit mapping the lifted state to observations,
#  y_t = C z_t + d
#Y = X.reshape((num_traj*steps_per_traj, -1))
#Z = np.concatenate((Z, np.ones((Z.shape[0],1))), axis=1)
#Y = np.concatenate((Y, np.ones((Y.shape[0],1))), axis=1)
#Cd, residuals, rank, s = np.linalg.lstsq(Y, Z, rcond=1)
#C = Cd[0:-1,0:-1]
#d = Cd[-1,0:-1]
#print("residual: ", np.linalg.norm(residuals))
#print(C.shape)
#print(d)

# Plot the eigenvalues of A
print("Plotting Eigenvalues")
eigvals = np.linalg.eigvals(A)
plt.figure()
plt.scatter(eigvals.real, eigvals.imag)

theta = np.linspace(0, 2*np.pi, 100)  # Unit circle
plt.plot(np.cos(theta), np.sin(theta), color='grey', linestyle='--')

plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.title("Eigenvalues of A")

# For comparison, plot the lifted state for a single trajectory
print("Making Plots")
plt.figure()

z = phi_X[0,:,:]  # The actual lifted state
z_pred = np.zeros_like(z)  # The predicted lifted state

y = X[0,:,:]  # The actual observation
y_pred = np.zeros_like(y)  # The predicted observation

z_pred[0,:] = z[0,:]  # Set the initial condition
#y_pred[0,:] = C@z_pred[0,:]
for step in range(1,steps_per_traj):
    z_pred[step,:] = z_pred[step-1,:] @ A
#    y_pred[step,:] = C @ z_pred[step,:] + d

#for i in range(3):
#    plt.subplot(3,1,i+1)
#    plt.plot(y[:,i], label="actual")
#    plt.plot(y_pred[:,i], label="predicted")
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
