#!/usr/bin/env python

##
#
# Train an RL agent using the simple REINFORCE algorithm with continuous action
# spaces.
#
##

import torch
import torch.nn as nn
from torch.distributions import Normal

import numpy as np


import gymnasium as gym

class PolicyNetwork(nn.Module):
    """
    The policy network takes in observations and outputs the mean and standard
    deviation of the action distribution.
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
            nn.Linear(64, 64),
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
        action = torch.normal(mean, std)
        return action, self.log_prob(mean, std, action)
    
    def log_prob(self, mean, std, action):
        """
        Given the mean and standard deviation of the action distribution and an
        action, calculate the log probability of the action.

        Args:
            mean: The mean of the action distribution
            std: The standard deviation of the action distribution
            action: The action

        Returns:
            The log probability of the action
        """
        distribution = Normal(mean, std)
        return distribution.log_prob(action).sum()
    
def reinforce(env, policy, num_episodes=1000, gamma=0.99, learning_rate=0.001):
    """
    Train the policy using the simple policy gradient REINFORCE algorithm.

    Args:
        env: The environment to train the policy on.
        policy: The policy to train.
        num_episodes: The number of episodes to train for.
        gamma: The discount factor.
        learning_rate: The learning rate.
    """
    # Define the optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

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

            # If the episode is done, calculate the return (sum of discounted rewards)
            if done:
                print(rewards)

                sys.exit()
                ## Calculate the returns
                #returns = calculate_returns(rewards, gamma)

                ## Calculate the discounted rewards
                #discounted_rewards = calculate_discounted_rewards(rewards, gamma)

    #    # Calculate the loss
    #    loss = calculate_loss(log_probs, discounted_rewards)

    #    # Zero the gradients
    #    optimizer.zero_grad()

    #    # Backpropagate the loss
    #    loss.backward()

    #    # Update the parameters
    #    optimizer.step()

    #    # Print the episode rewards
    #    print(f"Episode {episode + 1} reward: {sum(rewards)}")

def calculate_returns(rewards, gamma):
    """
    Compute the returns (sum of discounted rewards for each timestep), 

        G_t = \sum_{k=0}^{t} \gamma^k R_{k}.
    
    Args:
        rewards: The rewards for each timestep.
        gamma: The discount factor.
    
    Returns:
        The returns for each timestep.
    """
    returns = []
    for t in range(len(rewards)):
        returns.append(sum([gamma**k * rewards[k] for k in range(0, t+1)]))
    return returns

if __name__=="__main__":
    # Create the environment
    env = gym.make("Pendulum-v1")

    # Create the policy
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0])

    # Train the policy
    #reinforce(env, policy)