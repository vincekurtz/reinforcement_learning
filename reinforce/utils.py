import numpy as np
import matplotlib.pyplot as plt

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
        mean_rewards = np.array([np.mean(rewards) for rewards in self.rewards])
        std_rewards = np.array([np.std(rewards) for rewards in self.rewards])

        plt.plot(self.iteration_numbers, mean_rewards)
        plt.fill_between(self.iteration_numbers,
                         mean_rewards - std_rewards,
                         mean_rewards + std_rewards,
                         alpha=0.2)

        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        plt.show()
