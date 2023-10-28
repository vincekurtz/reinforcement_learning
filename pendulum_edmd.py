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

from solution_plots_pendulum import plot_pendulum_vector_field, plot_value_function

import torch
import numpy as np
import matplotlib.pyplot as plt

def gather_edmd_data(model, env, num_traj=10, steps_per_traj=100):
    """
    Gather data for fitting an EDMD model,

        z_{t+1} = A z_t
        y_t = C z_t

    Args:
        model: A stable-baselines3 PPO model that includes the lifting function
        env: A gym environment for the pendulum
        num_traj: The number of trajectories to simulate
        steps_per_traj: The number of steps to run each trajectory for.

    Return:
        Y: A numpy array of shape (num_traj, steps_per_traj, 3) containing the
            observations from each trajectory.
        Z: A numpy array of shape (num_traj, steps_per_traj, lifting_size)
            containing the lifted states from each trajectory.
    """
    # Allocate observation array
    Y = np.zeros((num_traj, steps_per_traj, 3))
    for traj in range(num_traj):
        # Reset to a new initial state
        obs, _ = env.reset()

        for step in range(steps_per_traj):
            # Store the current observation [cos(theta), sin(theta), theta_dot]
            Y[traj, step, :] = obs

            # Step the environment
            action, _ = model.predict(obs)
            obs, _, _, _, _ = env.step(action)

    # Compute phi(x) for each observation in the dataset
    phi = model.policy.mlp_extractor.lifting_function
    with torch.no_grad():
        Y_torch = torch.from_numpy(X).float().to(model.device)
        Z = phi(X_torch).cpu().numpy()

    return Y, Z

def perform_edmd(Y, Z):
    """
    Fit an EDMD model of the form

        z_{t+1} = A z_t
        y_t = C z_t
    
    to the data. 

    Args:
        Y: A numpy array of shape (num_traj, steps_per_traj, 3) containing the
            observations from each trajectory.
        Z: A numpy array of shape (num_traj, steps_per_traj, lifting_size)
            containing the lifted states from each trajectory.

    Returns:
        A: the dynamics matrix
        C: the projection matrix
    """
    # Fit the lifted dynamics as
    #  z_{t+1} = z_t A  (note the awkward order to fit with numpy least squares)
    Z = Z.reshape((num_traj*steps_per_traj, -1))
    Z_now = Z[0:-1,:]
    Z_next = Z[1:,:]
    A, residuals, rank, s = np.linalg.lstsq(Z_now, Z_next, rcond=1)
    print("dynamics residual: ", np.linalg.norm(residuals))

    # Compute a linear least-squares fit mapping the lifted state to observations,
    #  y_t = z_t C
    Y = Y.reshape((num_traj*steps_per_traj, -1))
    C, residuals, rank, s = np.linalg.lstsq(Z, Y, rcond=1)
    print("output residual: ", np.linalg.norm(residuals))

    return A, C

# Plot some vector fields 
def plot_controlled_pendulum_vector_field(env, model, n=25):
    """
    Make a vector field plot of the controlled pendulum dynamics.

    Args:
        env: A gym environment for the pendulum
        model: A stable-baselines3 PPO model for the controller
        n: The number of points to plot in each dimension
    """
    env.reset()

    thetas = np.linspace(-np.pi, 2*np.pi, n)
    theta_dots = np.linspace(-8, 8, n)

    for theta in thetas:
        for theta_dot in theta_dots:
            env.unwrapped.state = np.array([theta, theta_dot])
            obs = np.array([np.cos(theta), np.sin(theta), theta_dot])
            action, _ = model.predict(obs)
            env.step(action)
            dtheta = (env.unwrapped.state[0] - theta)
            dtheta_dot = (env.unwrapped.state[1] - theta_dot)

            plt.arrow(theta, theta_dot, dtheta, dtheta_dot,
                    head_width=0.05, head_length=0.1, color='blue', alpha=0.5)

    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    plt.xlim([-np.pi, 2*np.pi])
    plt.ylim([-8, 8])

if __name__=="__main__":
    # Try (vainly) to make things deterministic
    SEED = 1
    np.random.seed(SEED)
    set_random_seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Create the environment
    env = gym.make("Pendulum-v1")
    env.action_space.seed(SEED)

    # Load the learned model
    model = PPO.load("trained_models/pendulum")
        
    # Make vector field plots
    plt.subplot(2,2,1)
    plt.title("Uncontrolled Vector Field")
    plot_pendulum_vector_field()

    plt.subplot(2,2,3)
    plt.title("Controlled Vector Field")
    plot_controlled_pendulum_vector_field(env, model)

    plt.subplot(2,2,2)
    plt.title("Value Function (Quadratic in Koopman State))")
    plot_value_function(model)

    plt.subplot(2,2,4)
    plt.title("Koopman Model of Controlled Vector Field")

    plt.show()

## Plot the eigenvalues of A
#print("Plotting Eigenvalues")
#eigvals = np.linalg.eigvals(A)
#plt.figure()
#plt.scatter(eigvals.real, eigvals.imag)
#
#theta = np.linspace(0, 2*np.pi, 100)  # Unit circle
#plt.plot(np.cos(theta), np.sin(theta), color='grey', linestyle='--')
#
#plt.xlabel("Real Part")
#plt.ylabel("Imaginary Part")
#plt.title("Eigenvalues of A")

## For comparison, plot the lifted state for a single trajectory
#print("Making Plots")
#plt.figure()
#
#z = phi_X[0,:,:]  # The actual lifted state
#z_pred = np.zeros_like(z)  # The predicted lifted state
#
#y = X[0,:,:]  # The actual observation
#y_pred = np.zeros_like(y)  # The predicted observation
#
#z_pred[0,:] = z[0,:]  # Set the initial condition
#y_pred[0,:] = z_pred[0,:] @ C
#for step in range(1,steps_per_traj):
#    z_pred[step,:] = z_pred[step-1,:] @ A
#    y_pred[step,:] = z_pred[step,:] @ C
#
#for i in range(3):
#    plt.subplot(3,1,i+1)
#    plt.plot(y[:,i], label="actual")
#    plt.plot(y_pred[:,i], label="predicted")
##for i in range(25):
##    plt.subplot(5,5,i+1)
##    plt.plot(z[:,i], label="actual")
##    plt.plot(z_pred[:,i], label="predicted")
#
#plt.legend()
#plt.show()

## Plot observations over time
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
