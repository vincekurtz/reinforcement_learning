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
    print("Gathering Data")

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
        Y_torch = torch.from_numpy(Y).float().to(model.device)
        Z = phi(Y_torch).cpu().numpy()

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
    print("Performing EDMD")

    assert Y.shape[0] == Z.shape[0]
    assert Y.shape[1] == Z.shape[1]
    num_traj = Y.shape[0]
    steps_per_traj = Y.shape[1]

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

def plot_eigenvalues(A):
    """
    Plot the eigenvalues of the discrete-time state update matrix A, and overlay
    the unit circle.
    """
    plt.figure()

    eigvals = np.linalg.eigvals(A)
    plt.scatter(eigvals.real, eigvals.imag)

    theta = np.linspace(0, 2*np.pi, 100)  # Unit circle
    plt.plot(np.cos(theta), np.sin(theta), color='grey', linestyle='--')

    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title("Eigenvalues of A")

def compare_lifted_state_trajectories(env, model, A, num_steps=100):
    """
    Compare the actual lifted state z_t = phi(y_t) along a trajectory to the
    predicted lifted state from the learned Koopman model.

    Args:
        env: A gym environment for the pendulum
        model: A stable-baselines3 PPO model for the controller + lifting function
        A: The learned Koopman matrix
        num_steps: The number of steps to simulate
    """
    plt.figure()

    # Get the lifting function
    phi = model.policy.mlp_extractor.lifting_function

    # Simulate a trajectory closed-loop
    obs, _ = env.reset()
    Y = np.zeros((num_steps, 3))
    for step in range(num_steps):
        # Record the observation
        Y[step,:] = obs

        # Choose an action according to the learned policy
        action, _ = model.predict(obs)

        # Step the environment
        obs, _, _, _, _ = env.step(action)

    # Compute the lifted state along the trajectory
    with torch.no_grad():
        Y_torch = torch.from_numpy(Y).float().to(model.device)
        Z = phi(Y_torch).cpu().numpy()

    # Simulate the linear Koopman model
    Z_pred = np.zeros_like(Z)
    Z_pred[0,:] = Z[0,:]
    for step in range(1,num_steps):
        Z_pred[step,:] = Z_pred[step-1,:] @ A

    # Plot the first 32 dimensions of the lifted state
    for i in range(32):
        plt.subplot(4,8,i+1)
        plt.plot(Z[:,i], label="actual")
        plt.plot(Z_pred[:,i], label="predicted")
    plt.legend()

def compare_trajectories(env, model, A, C, num_steps=100):
    """
    Compare a trajectory we get from simulating the controlled system to a
    trajectories we get from the learned Koopman model of the controlled system. 

    Args:
        env: A gym environment for the pendulum
        model: A stable-baselines3 PPO model for the controller + lifting function
        A: The learned Koopman matrix
        C: The learned mapping from lifted state to observation
        num_steps: The number of steps to simulate
    """
    plt.figure()

    # Get the lifting function
    phi = model.policy.mlp_extractor.lifting_function

    # Simulate a trajectory closed-loop
    obs, _ = env.reset()
    Y = np.zeros((num_steps, 3))
    for step in range(num_steps):
        Y[step,:] = obs
        action, _ = model.predict(obs)
        obs, _, _, _, _ = env.step(action)

    # Compute the lifted state at the first timestep
    with torch.no_grad():
        y0_torch = torch.from_numpy(Y[0,:]).float().to(model.device)
        z0 = phi(y0_torch).cpu().numpy()

    # Simulate the linear Koopman model
    nz = z0.shape[0]
    Z = np.zeros((num_steps, nz))
    Z[0,:] = z0
    Y_pred = np.zeros((num_steps, 3))
    for step in range(0,num_steps-1):
        Y_pred[step,:] = Z[step,:] @ C
        Z[step+1,:] = Z[step,:] @ A
    Y_pred[-1,:] = Z[-1,:] @ C

    # Plot the observations
    plt.subplot(3,1,1)
    plt.plot(Y[:,0], label="actual")
    plt.plot(Y_pred[:,0], label="predicted")
    plt.ylabel("cos(theta)")
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(Y[:,1], label="actual")
    plt.plot(Y_pred[:,1], label="predicted")
    plt.ylabel("sin(theta)")

    plt.subplot(3,1,3)
    plt.plot(Y[:,2], label="actual")
    plt.plot(Y_pred[:,2], label="predicted")
    plt.ylabel("theta_dot")

def plot_koopman_vector_field(model, A, C, n=10):
    """
    Plot the vector field of the learned Koopman model of the controlled
    pendulum dynamics.

    Args:
        model: A stable-baselines3 PPO model for the controller + lifting function
        A: The learned Koopman matrix
        C: The learned mapping from lifted state to observation
        n: The number of points to plot in each dimension
    """
    # Get the lifting function
    phi = model.policy.mlp_extractor.lifting_function

    for _ in range(10):
        # Sample an initial state
        theta = np.random.uniform(-np.pi, 2*np.pi)
        theta_dot = np.random.uniform(-8, 8)
        y0 = np.array([np.cos(theta), np.sin(theta), theta_dot])

        # Compute the lifted state at the first timestep
        with torch.no_grad():
            y0_torch = torch.from_numpy(y0).float().to(model.device)
            z0 = phi(y0_torch).cpu().numpy()

        # Simulate the linear Koopman model
        num_steps = 100
        nz = z0.shape[0]
        Z = np.zeros((num_steps, nz))
        Z[0,:] = z0
        Y_pred = np.zeros((num_steps, 3))
        for step in range(0,num_steps-1):
            Y_pred[step,:] = Z[step,:] @ C
            Z[step+1,:] = Z[step,:] @ A
        Y_pred[-1,:] = Z[-1,:] @ C

        old_theta = np.arctan2(Y_pred[0,1], Y_pred[0,0])
        old_theta_dot = Y_pred[0,2]
        for t in range(num_steps):
            cos_theta = Y_pred[t,0]
            sin_theta = Y_pred[t,1]
            new_theta_dot = Y_pred[t,2]

            # Convert cos(theta), sin(theta) to theta. This is a little tricky
            # because arctan2 only returns values in [-pi, pi], so we have to
            # adjust to avoid wrapping
            new_theta = np.arctan2(sin_theta, cos_theta)
            if new_theta < old_theta - np.pi:
                new_theta += 2*np.pi
            elif new_theta > old_theta + np.pi:
                new_theta -= 2*np.pi

            plt.arrow(old_theta, old_theta_dot, new_theta-old_theta, new_theta_dot-old_theta_dot,
                    head_width=0.05, head_length=0.1, color='blue', alpha=0.5)

            old_theta = new_theta
            old_theta_dot = new_theta_dot

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

    # Gather data for EDMD
    Y, Z = gather_edmd_data(model, env, num_traj=100, steps_per_traj=100)

    # Fit an EDMD model
    A, C = perform_edmd(Y, Z)

    ## Compare predictions in the lifted space
    #compare_lifted_state_trajectories(env, model, A, num_steps=100)

    ## Compare predictions in the observation space
    #compare_trajectories(env, model, A, C, num_steps=100)

    ## Plot the eigenvalues of the learned Koopman operator approximation
    #plot_eigenvalues(A)
        
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
    plot_koopman_vector_field(model, A, C)

    plt.show()
