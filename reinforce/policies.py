import torch
import torch.nn as nn
from torch.distributions import Normal

class MlpPolicy(nn.Module):
    """
    A simple policy network based on a Multilayer Perceptron (MLP).

    Takes in observations and outputs the mean and standard deviation of an
    action distribution.
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
    