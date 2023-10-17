#!/usr/bin/env python

##
#
# Train an RL agent using the simple REINFORCE algorithm with continuous action
# spaces.
#
##

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import time
import matplotlib.pyplot as plt
import pickle

# Set random seed for reproducability
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True

class PolicyNetwork(nn.Module):
    """
    The policy network takes in observations and outputs the mean and standard
    deviation of an action distribution.
    """
    def __init__(self, input_size, output_size) -> None:
        """
        Initialize the policy network.
        
        Args:
            input_size: The size of the observation space
            output_size: The size of the action space
        """
        super().__init__()

        # Define the mean network
        self.mean_network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

        # Define a single parameter for the (log) standard deviation
        self.log_std = nn.Parameter(torch.zeros(output_size), requires_grad=True)

    def forward(self, x):
        """
        Forward pass through the policy network.

        Args:
            x: The observation
        
        Returns:
            The mean and standard deviation of the action distribution
        """
        mean = self.mean_network(x)
        std = torch.exp(self.log_std)
        return mean, std
    
    def sample(self, x):
        """
        Given an observation, sample an action from the action distribution.

        Args:
            x: The observation

        Returns:
            The action and the log probability of the action
        """
        mean, std = self.forward(x)
        distribution = Normal(mean, std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum()
        return action, log_prob
    
class ConvergencePlotter:
    """
    A little struct for holding convergence information and tools for making
    cute plots of it.
    """
    def __init__(self):
        # List of list of rewards at each iteration
        self.rewards = []

        # List of iteration numbers, since we'll only add to this every so often
        self.iteration_numbers = []

    def add(self, iteration_number, rewards):
        self.rewards.append(rewards)
        self.iteration_numbers.append(iteration_number)

    def plot(self):
        """
        Make a matplotlib plot of the convergence information.
        """
        mean_rewards = [np.mean(rewards) for rewards in self.rewards]
        std_rewards = [np.std(rewards) for rewards in self.rewards]
        plt.errorbar(self.iteration_numbers, mean_rewards, yerr=std_rewards)
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        plt.show()

@torch.no_grad() 
def evaluate_policy(env, policy, num_episodes=100):
    """
    Test the policy by running it for a number of episodes.

    Args:
        env: The environment to test on.
        policy: The policy to test.
        num_episodes: The number of episodes to run.

    Returns:
        A list of total rewards for each episode
    """
    rewards = [0 for _ in range(num_episodes)]
    for episode in range(num_episodes):
        obs, _ = env.reset()

        # Simulate the episode until the end
        done = False
        while not done:
            # We'll take the mean of the action distribution rather than sample
            action, _ = policy(torch.tensor(obs, dtype=torch.float32))
            obs, reward, terminated, truncated, _ = env.step(action.detach().numpy())
            done = terminated or truncated
            rewards[episode] += reward

    return rewards
    
def reinforce(env, policy, num_episodes=1000, gamma=0.99, learning_rate=0.001, print_interval=100, checkpoint_interval=1000):
    """
    Train the policy using the simple policy gradient algorithm REINFORCE.

    Args:
        env: The environment to train the policy on.
        policy: The policy to train.
        num_episodes: The number of episodes to train for.
        gamma: The discount factor.
        learning_rate: The learning rate.
        print_interval: How often to print a summary of how we're doing.
        checkpoint_interval: How often to save the policy to disk.
    """
    # Define the optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    # Store stuff for logging
    avg_rewards = []
    plotter = ConvergencePlotter()
    start_time = time.time()

    # Iterate over episodes
    for episode in range(num_episodes):
        # Reset stored rewards and log probabilities
        rewards = []
        log_probs = []

        # Reset the environment and get the initial observation
        observation, info = env.reset()

        # Iterate until the episode is done
        done = False
        while not done:
            # Get the action from the policy
            action, log_prob = policy.sample(torch.tensor(observation, dtype=torch.float32))

            # Apply the action to the environment
            observation, reward, terminated, truncated, info = env.step(action.detach().numpy())
            done = terminated or truncated

            # Record the resulting rewards and log probabilities
            rewards.append(reward)
            log_probs.append(log_prob)

        # Once the episode is over, calculate the loss, 
        #   J = -1/T * sum(log_prob * G_t),
        # where G_t = sum(gamma^k * r_{t+k}) is the discounted return.
        T = len(log_probs)
        returns = np.zeros(T)
        returns[-1] = rewards[-1]
        for t in range(T-2, -1, -1):
            returns[t] = rewards[t] + gamma * returns[t+1]

        loss = -1/T * sum([log_probs[t] * returns[t] for t in range(T)])

        # Compute gradients and update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the average reward every so often
        avg_rewards.append(sum(rewards))
        if episode % print_interval == 0:
            print(f"Episode {episode}, avg reward: {np.mean(avg_rewards)}, time_elapsed: {time.time() - start_time:.2f}")
            avg_rewards = []

        # Save the policy every so often
        if episode > 0 and episode % checkpoint_interval == 0:
            # Do some more intense evaluation
            rewards = evaluate_policy(env, policy, num_episodes=100)
            print(f"Average reward: {np.mean(rewards)}. Std dev: {np.std(rewards)}")

            # Save this data for plotting later
            plotter.add(episode, rewards)

            # Save the policy
            fname = f"checkpoints/policy_{episode}.pt"
            print(f"Saving checkpoint to {fname}")
            torch.save(policy.state_dict(), fname)

    # Save a pickle of the plotter in case we want to look at it later
    with open("plotter.pkl", "wb") as f:
        pickle.dump(plotter, f)

    # Make plots of the convergence
    plotter.plot()

if __name__=="__main__":
    # Create the environment
    env = gym.make("Pendulum-v1")
    env.reset(seed=SEED)

    # Create the policy
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0])
    #policy.load_state_dict(torch.load('policy.pt'))

    # Train the policy
    reinforce(env, policy, num_episodes=60000, learning_rate=5e-4)

    # Save the policy
    torch.save(policy.state_dict(), "policy.pt")
