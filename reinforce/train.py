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
import time
import pickle

from policies import MlpPolicy, RnnPolicy, KoopmanPolicy
from utils import ConvergencePlotter

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
        policy.reset()

        # Simulate the episode until the end
        done = False
        while not done:
            # We'll take the mean of the action distribution rather than sample
            action, _ = policy(torch.tensor(obs, dtype=torch.float32))
            obs, reward, terminated, truncated, _ = env.step(action.detach().numpy())
            done = terminated or truncated
            rewards[episode] += reward

    return rewards
    
def reinforce(env, policy, num_episodes=1000, gamma=0.99, learning_rate=0.001, print_interval=10, checkpoint_interval=100):
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

        # Reset the hidden state of the policy network, if necessary
        policy.reset()

        # Iterate until the episode is done
        max_steps = 500
        for t in range(max_steps):
            # Get the action from the policy
            action, log_prob = policy.sample(torch.tensor(observation, dtype=torch.float32))

            # Apply the action to the environment
            observation, reward, terminated, truncated, info = env.step(action.detach().numpy())

            # Record the resulting rewards and log probabilities
            rewards.append(reward)
            log_probs.append(log_prob)

            if terminated or truncated:
                break

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
    
    # Save the policy
    torch.save(policy.state_dict(), "policy.pt")

    # Save a pickle of the plotter in case we want to look at it later
    with open("plotter.pkl", "wb") as f:
        pickle.dump(plotter, f)

    # Make plots of the convergence
    plotter.plot()

if __name__=="__main__":
    # Set random seed for reproducability
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True

    # Create the environment
    #env = gym.make("Pendulum-v1")
    env = gym.make("InvertedPendulum-v4")
    env.reset(seed=SEED)

    # Create the policy
    #policy = MlpPolicy(env.observation_space, env.action_space)
    #policy = RnnPolicy(env.observation_space, env.action_space)
    policy = KoopmanPolicy(env.observation_space, env.action_space)

    # Train the policy
    reinforce(env, policy, num_episodes=3000, learning_rate=1e-3)
