import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    """
    Abstract base class for a policy network that takes continuous observations
    and outputs continuous actions.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()

        # Check that observation and action spaces are continuous
        assert isinstance(observation_space, gym.spaces.Box)
        assert isinstance(action_space, gym.spaces.Box)

        self.input_size = observation_space.shape[0]
        self.output_size = action_space.shape[0]

    def forward(self, x):
        """
        Forward pass through the policy network.

        Args:
            x: The observation
        
        Returns:
            The mean and standard deviation of the action distribution
        """
        raise NotImplementedError

    def sample(self, x):
        """
        Given an observation, sample an action from the action distribution.

        Args:
            x: The observation

        Returns:
            The action and the log probability of the action
        """
        raise NotImplementedError
    
    def reset(self):
        """
        Reset the hidden state, e.g. between episodes, if necessary.
        """
        pass

class MlpPolicy(PolicyNetwork):
    """
    A simple policy network based on a Multilayer Perceptron (MLP).
    """
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # Define the mean network
        self.mean_network = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )

        # Define a single parameter for the (log) standard deviation
        self.log_std = nn.Parameter(torch.zeros(self.output_size), requires_grad=True)

    def forward(self, x):
        mean = self.mean_network(x)
        std = torch.exp(self.log_std)
        return mean, std
    
    def sample(self, x):
        mean, std = self.forward(x)
        distribution = Normal(mean, std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum()
        return action, log_prob
    