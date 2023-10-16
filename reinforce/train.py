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
    
def reinforce(env, policy, num_episodes=1000, gamma=0.99, learning_rate=0.001):
    """
    Train the policy using the simple policy gradient algorithm REINFORCE.

    Args:
        env: The environment to train the policy on.
        policy: The policy to train.
        num_episodes: The number of episodes to train for.
        gamma: The discount factor.
        learning_rate: The learning rate.
    """
    # Define the optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    # Store stuff for logging
    avg_rewards = []
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

        # Print the average reward every 100 episodes
        avg_rewards.append(sum(rewards))
        if episode % 100 == 0:
            print(f"Episode {episode + 1}, avg reward: {np.mean(avg_rewards)}, time_elapsed: {time.time() - start_time:.2f}")
            avg_rewards = []

if __name__=="__main__":
    # Create the environment
    env = gym.make("Pendulum-v1")
    env.reset(seed=SEED)

    # Create the policy
    policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0])

    # Train the policy
    reinforce(env, policy, num_episodes=100000)

    # Save the policy
    torch.save(policy.state_dict(), "policy.pt")
