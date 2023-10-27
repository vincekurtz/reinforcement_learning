import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

class Quadratic(nn.Module):
    """
    A simple quadratic function

        y = x'Ax + b'x + c

    where A, b, and c are learnable parameters.
    """
    def __init__(self, input_size):
        super().__init__()
        self.A = nn.Parameter(torch.randn(input_size, input_size), requires_grad=True)
        self.b = nn.Parameter(torch.randn(input_size), requires_grad=True)
        self.c = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        y = (x @ self.A @ x.T).diag() + x @ self.b + self.c
        return y.unsqueeze(-1)

class KoopmanMlpExtractor(nn.Module):
    """
    A custom neural net for both the policy and the value function.
    """
    def __init__(self, input_size, output_size, num_linear_systems):
        super().__init__()

        # The custom network must have these output dimensions as attributes
        # with these names. The PPO implementation adds an additional linear
        # layer that maps from 'latent_dim_pi' to actions and from
        # 'latent_dim_vf' to values
        self.latent_dim_pi = output_size
        self.latent_dim_vf = 1

        # Policy is piecewise linear with a learned switching surface
        self.linear_systems = nn.ModuleList([
            nn.Linear(input_size, output_size, bias=False)
            for _ in range(num_linear_systems)])
        
        # Value function is piecewise quadratic with the same switching surface
        self.quadratic_systems = nn.ModuleList([
            Quadratic(input_size)
            for _ in range(num_linear_systems)])

        # We define a switching surface between regimes with a neural net. This
        # network outputs scores in [0,1] that determine which linear layer is active
        #chooser_network_hidden_size = 64
        #self.chooser = nn.Sequential(
        #    nn.Linear(input_size, chooser_network_hidden_size), nn.ReLU(),
        #    nn.Linear(chooser_network_hidden_size, num_linear_systems),
        #    nn.Sigmoid())
        self.chooser = nn.Sequential(
            nn.Linear(input_size, num_linear_systems),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)

    def forward_actor(self, x):
        # Compute the switching coefficients
        sigma = self.chooser(x)   # shape: [batch_size, num_linear_systems]

        # Pass input through each linear system
        output = [linear(x) for linear in self.linear_systems]
        output = torch.stack(output, dim=1)  # shape: [batch_size, num_linear_systems, output_size]

        # Weight the outputs by the switching coefficients
        output = (sigma.unsqueeze(-1) * output).sum(dim=1)  # shape: [batch_size, output_size]

        # Normalize output so that weighting coefficients sum to 1
        #output /= sigma.sum(dim=1, keepdim=True)

        return output

    def forward_critic(self, x):
        # Compute the switching coefficients
        sigma = self.chooser(x)   # shape: [batch_size, num_linear_systems]

        # Pass input through each quadratic value approximator
        output = [quadratic(x) for quadratic in self.quadratic_systems]
        output = torch.stack(output, dim=1)  # shape: [batch_size, num_linear_systems, 1]

        # Weight the output by the switching coefficients
        output = (sigma.unsqueeze(-1) * output).sum(dim=1)  # shape: [batch_size, 1]

        return output

class KoopmanPolicy(ActorCriticPolicy):
    """
    A custom actor-critic policy that uses our custom Koopman architecture.
    Structured so that the SB3 PPO implementation can use our architecture
    directly.
    """
    def __init__(self, observation_space, action_space, lr_schedule,
                 num_linear_systems, *args, **kwargs):
        self.num_linear_systems = num_linear_systems
        super().__init__(observation_space, action_space, lr_schedule, *args,
                **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = KoopmanMlpExtractor(
            self.features_dim, self.action_space.shape[0], 
            self.num_linear_systems)
        
    def save(self, path):
        """
        Save the policy to disk, including some extra custom params. 
        """
        super().save(path, include=["num_linear_systems"])

class ParallelLinear(nn.Module):
    """
    Compute the weighted sum of multiple linear blocks in parallel, i.e.,

        y = K1*x + K2*x + ... + Kn*x

    """
    def __init__(self, input_size, output_size, num_blocks=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size, output_size, bias=False) 
            for _ in range(num_blocks)])
        
    def forward(self, x):
        # Sum the output of each linear layer
        output = [layer(x) for layer in self.layers]
        output = torch.stack(output, dim=1)
        output = output.sum(dim=1)
        return output
    
class SeriesLinear(nn.Module):
    """
    Compute the output of multiple linear layers connected in series, i.e.,
        
       y = Kn * ... * K2 * K1 * x
    
    """
    def __init__(self, input_size, output_size, num_blocks=2):
        """
        Args:
            input_size: The dimension of the input
            output_size: The dimension of the output
            hidden_sizes: A list of hidden layer sizes
        """
        super().__init__()

        output_sizes = [output_size]*num_blocks
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, output_sizes[0], bias=False))
        for i in range(1, len(output_sizes)):
            self.layers.append(
                nn.Linear(output_sizes[i-1], output_sizes[i], bias=False))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class HierarchyLinear(nn.Module):
    """
    Compute the output of multiple linear layers connected in a hierarchical fashion, where
    the output of the ith layer is fed as an input to the (i-1)th layer. 
    """
    def __init__(self, input_size, output_size, num_blocks=2):
        super().__init__()

        output_sizes = num_blocks*[output_size]
        self.layers = nn.ModuleList()

        for i in range(len(output_sizes)-1):
            # Most layers have an input from the previous layer and an input from the
            # the global input
            self.layers.append(
                nn.Linear(input_size + output_sizes[i+1], output_sizes[i], bias=False))
        # The final layer only has an input from the global input
        self.layers.append(
            nn.Linear(input_size, output_sizes[-1], bias=False))
        
    def forward(self, x):
        y = self.layers[-1](x)
        for layer in reversed(self.layers[:-1]):
            y = layer(torch.cat((x, y), dim=1))
        return y

class LinearMlpExtractor(nn.Module):
    """
    A neural net containing both the policy and the value function. The policy
    is a linear map from observations to actions, and the value function is
    quadratic.
    """
    def __init__(self, input_size, output_size):
        super().__init__()

        # The custom network must have these output dimensions as attributes
        # with these names. The PPO implementation adds an additional linear
        # layer that maps from 'latent_dim_pi' to actions and from
        # 'latent_dim_vf' to values
        self.latent_dim_pi = output_size
        self.latent_dim_vf = 1

        # Policy
        #self.policy_network = nn.Linear(input_size, output_size, bias=False)
        #self.policy_network = ParallelLinear(input_size, output_size, num_blocks=1)
        #self.policy_network = SeriesLinear(input_size, output_size, num_blocks=1)
        self.policy_network = HierarchyLinear(input_size, output_size, num_blocks=1)

        # Value function
        #self.value_network = Quadratic(input_size)
        self.value_network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)

    def forward_actor(self, x):
        return self.policy_network(x)

    def forward_critic(self, x):
        return self.value_network(x)

class LinearPolicy(ActorCriticPolicy):
    """
    A simple linear policy mapping observations to actions.
    """
    def __init__(self, observation_space, action_space, lr_schedule, *args, 
                 **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args,
                **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = LinearMlpExtractor(
            self.features_dim, self.action_space.shape[0])
        