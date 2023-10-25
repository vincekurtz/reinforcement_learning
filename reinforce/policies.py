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
        mean, std = self.forward(x)
        distribution = Normal(mean, std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum()
        return action, log_prob
    
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
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )

        # Define a single parameter for the (log) standard deviation
        self.log_std = nn.Parameter(torch.zeros(self.output_size), requires_grad=True)

    def forward(self, x):
        mean = self.mean_network(x)
        std = torch.exp(self.log_std)
        return mean, std
    
class RnnPolicy(PolicyNetwork):
    """
    A simple policy network based on a Recurrent Neural Network (RNN).
    """
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # Define the mean network as an RNN
        self.state_size = 8  # Size of the hidden state
        self.recurrent_network = nn.RNN(self.input_size, self.state_size, nonlinearity='tanh', batch_first=True)
        self.output_network = nn.Linear(self.state_size, self.output_size)

        # Define a single parameter for the (log) standard deviation
        self.log_std = nn.Parameter(torch.zeros(self.output_size), requires_grad=True)

        # Reset the hidden state
        self.reset()

    def forward(self, x):
        _, self.hidden_state = self.recurrent_network(x.unsqueeze(0), self.hidden_state)
        mean = self.output_network(self.hidden_state.squeeze(0))
        std = torch.exp(self.log_std)
        return mean, std
    
    def reset(self):
        self.hidden_state = torch.zeros(1, self.state_size)

class KoopmanPolicy(PolicyNetwork):
    """
    A recurrent policy network based on Koopman theory. The controller is
    treated as a linear system with lifted input

        z_{t+1} = Az_t + Bϕ(u_t)
        y_t = Cz_t,

    where u_t is the input (observations), y_t is the output (actions), and z_t
    is the state. Koopman tells us that any nonlinear system can be represented 
    as a linear system in an infinite-dimensional space, and the perfect
    controller can be described as a nonlinear system, so we'll learn a
    finite-dimensional linear approximation to the perfect controller.
    """
    def __init__(self, observation_space, action_space, lifted_state_size=4):
        super().__init__(observation_space, action_space)

        # Define the mean network as a linear system
        self.linear_system = LiftedInputLinearSystem(self.input_size, self.output_size, lifted_state_size)

        # Define a single parameter for the (log) standard deviation
        self.log_std = nn.Parameter(torch.zeros(self.output_size), requires_grad=True)

    def forward(self, obs):
        mean = self.linear_system(obs)
        std = torch.exp(self.log_std)
        return mean, std
    
    def reset(self):
        self.linear_system.reset()

class LiftedInputLinearSystem(nn.Module):
    """
    A linear system of the form

        z_{t+1} = Az_t + Bϕ(u_t) 
        y_t = Cz_t,

    Where u_t is the input (observations), y_t is the output (actions), and z_t
    is the state. The input is lifted via an MLP ϕ.
    """
    def __init__(self, input_size, output_size, state_size, lifted_input_size=None):
        super().__init__()
        if lifted_input_size is None:
            lifted_input_size = state_size
        self.state_size = state_size

        # Linear system matrices
        self.A = nn.Linear(state_size, state_size, bias=False)
        self.B = nn.Linear(lifted_input_size, state_size, bias=False)
        self.C = nn.Linear(state_size, output_size, bias=False)

        # MLP for lifting the input
        hidden_size = lifted_input_size
        self.phi = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, lifted_input_size)
        )

        # Allocate the state
        self.reset()

    def forward(self, u):
        self.z = self.A(self.z) + self.B(self.phi(u))
        y = self.C(self.z)
        return y
    
    def reset(self):
        self.z = torch.zeros(self.state_size)

class LinearSystem(nn.Module):
    """
    A simple linear system (a.k.a. recurrent network) of the form
    
            x_{t+1} = Ax_t + Bu_t,
            y_t = Cx_t
    
    where u_t is the input (observations), y_t is the output (actions), and x_t
    is the (hidden) state. 
    """
    def __init__(self, input_size, output_size, state_size):
        super().__init__()
        self.state_size = state_size

        # Linear system matrices
        self.A = nn.Linear(state_size, state_size, bias=False)
        self.B = nn.Linear(input_size, state_size, bias=False)
        self.C = nn.Linear(state_size, output_size, bias=False)

        # Allocate the state
        self.reset()

    def forward(self, u):
        self.x = self.A(self.x) + self.B(u)
        y = self.C(self.x)
        return y
    
    def reset(self):
        self.x = torch.zeros(self.state_size)

class TwoInputLinearSystem(nn.Module):
    """
    A linear system with two inputs, u = [u⁽¹⁾, u⁽²⁾], of the form

        x_{t+1} = Ax_t + Bu_t,
        y_t = Cx_t

    This is useful for modeling the interconnections of multiple linear systems.
    """
    def __init__(self, input1_size, input2_size, output_size, state_size):
        super().__init__()
        self.state_size = state_size

        # Linear system matrices
        self.A = nn.Linear(state_size, state_size, bias=False)
        self.B1 = nn.Linear(input1_size, state_size, bias=False)
        self.B2 = nn.Linear(input2_size, state_size, bias=False)
        self.C = nn.Linear(state_size, output_size, bias=False)

        # Allocate the state
        self.reset()

    def forward(self, u1, u2):
        self.x = self.A(self.x) + self.B1(u1) + self.B2(u2)
        y = self.C(self.x)
        return y
    
    def reset(self):
        self.x = torch.zeros(self.state_size)

class DeepKoopmanPolicy(PolicyNetwork):
    """
    A policy network composed of several interconnected linear systems.
    """
    def __init__(self, observation_space, action_space, state_sizes=[4, 4], output_sizes=[4]):
        """
        Args:
            observation_space: The observation space of the environment.
            action_space: The action space of the environment. state_sizes: A
            list of the sizes of the hidden state for each linear system block
            output_sizes: A list of the output size for each linear system block
        """
        super().__init__(observation_space, action_space)

        self.num_blocks = len(state_sizes)
        output_sizes.append(self.output_size)   # The last block's output is the action
        assert len(output_sizes) == self.num_blocks  

        # Define several interconnected linear systems that eventually output the mean
        self.linear_systems = nn.ModuleList()
        self.linear_systems.append(
            LinearSystem(self.input_size,  # The only input is the system observation
                         output_sizes[0],  # Output goes to the next linear system
                         state_sizes[0]))
        for i in range(1, self.num_blocks):
            self.linear_systems.append(
                TwoInputLinearSystem(
                    self.input_size,   # First input is the system observation
                    output_sizes[i-1], # Second input is from the previous linear system
                    output_sizes[i],   # Output goes to the next linear system
                    state_sizes[i]))

        # Define a single parameter for the (log) standard deviation
        self.log_std = nn.Parameter(torch.zeros(self.output_size), requires_grad=True)
      
    def forward(self, obs):
        # Compute the mean by passing through the linear systems
        mean = self.linear_systems[0](obs)
        for i in range(1, self.num_blocks):
            mean = self.linear_systems[i](obs, mean)

        # Standard deviation is a constant parameter
        std = torch.exp(self.log_std)
        return mean, std
    
    def reset(self):
        # Reset the state of each linear system
        for linear_system in self.linear_systems:
            linear_system.reset()

class KoopmanBilinearPolicy(PolicyNetwork):
    """
    A recurrent policy network based on Koopman eigenfunctions. The controller
    is a bilinear system,

        z_{t+1} = Λz_t + z_t*Bu_t,
        y_t = Cz_t,

    where z_t are Koopman eigenfunctions of the controlled system, u_t is the
    input (observations), y_t is the output (actions), and z_t are Koopman
    eigenfunctions. Λ is a diagonal matrix that stores the corresponding Koopman
    eigenvalues. Koopman theory tells us that (under some assumptions), any
    control-affine system can be written in this form.
    """
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # Decide on the size of the hidden state z
        self.hidden_state_size = 8

        # System matrices
        self.Lambda = nn.Parameter(torch.randn(self.hidden_state_size), requires_grad=True)
        self.B = nn.Bilinear(self.input_size, self.hidden_state_size, self.hidden_state_size, bias=False)
        self.C = nn.Linear(self.hidden_state_size, self.output_size, bias=False)

        # Define a single parameter for the (log) standard deviation
        self.log_std = nn.Parameter(torch.zeros(self.output_size), requires_grad=True)

        # Allocate the hidden state
        self.reset()

    def forward(self, u):
        # Compute the ouptut of the system (mean)
        self.z = self.Lambda * self.z + self.B(u, self.z)
        y = self.C(self.z)

        # Return the mean and standard deviation of the action distribution
        std = torch.exp(self.log_std)
        return y, std

    def reset(self):
        self.z = torch.zeros(self.hidden_state_size) 
