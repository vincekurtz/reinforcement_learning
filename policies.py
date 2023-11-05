import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

class Quadratic(nn.Module):
    """
    A simple quadratic function

        y = x'Qx + b'x + c

    where Q, b, and c are learnable parameters.
    """
    def __init__(self, input_size):
        super().__init__()
        self.Q = nn.Parameter(torch.randn(input_size, input_size), 
                              requires_grad=True)
        self.b = nn.Parameter(torch.randn(input_size), requires_grad=True)
        self.c = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        # Compute the output of the quadratic function [batch_size, 1]
        y = (x @ self.Q @ x.T).diag() + x @ self.b + self.c
        return y.unsqueeze(-1)
    
class PsdQuadratic(nn.Module):
    """
    A PSD quadratic function

        y = x'Qx + b'x + c

    where Q = LL' is positive semi-definite, L is a lower triangular matrix, 
    and (L, b, c) are learnable parameters
    """
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        # Parameterize Q = LL' with L a lower triangular matrix
        self.L = nn.Parameter(torch.randn(input_size, input_size), 
                              requires_grad=True)
        self.mask = torch.triu(torch.ones(input_size, input_size, dtype=bool), 
                               diagonal=1)
        with torch.no_grad():
            self.L.masked_fill_(self.mask, 0.0)

        self.b = nn.Parameter(torch.randn(input_size), requires_grad=True)
        self.c = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        # Mask out the zero elements of L
        with torch.no_grad():
            self.L.masked_fill_(self.mask, 0.0)

        # Compute the output of the quadratic function [batch_size, 1]
        y = (x @ self.L @ self.L.T @ x.T).diag() + x @ self.b + self.c
        return y.unsqueeze(-1)
    
class DiagonalQuadratic(nn.Module):
    """
    A simple quadratic function

        y = x'Qx + b'x + c

    where Q is a diagonal matrix, and (Q, b, c) are learnable parameters.
    """
    def __init__(self, input_size):
        super().__init__()
        self.Q = nn.Parameter(torch.randn(input_size), requires_grad=True)
        self.b = nn.Parameter(torch.randn(input_size), requires_grad=True)
        self.c = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        y = (x @ torch.diag(self.Q) @ x.T).diag() + x @ self.b + self.c
        return y.unsqueeze(-1)
    
class KoopmanMlpExtractor(nn.Module):
    """
    A custom neural net for both the policy and the value function.
    """
    def __init__(self, input_size, output_size, lifting_dim):
        super().__init__()

        # The custom network must have these output dimensions as attributes
        # with these names. The PPO implementation adds an additional linear
        # layer that maps from 'latent_dim_pi' to actions and from
        # 'latent_dim_vf' to values
        self.latent_dim_pi = output_size
        self.latent_dim_vf = 1

        # Lifting function maps to a higher-dimensional Koopman-invariant space
        self.phi = nn.Sequential(
                nn.Linear(input_size, lifting_dim), nn.GELU(),
                nn.Linear(lifting_dim, lifting_dim), nn.GELU())
        
        # Policy is a linear map from the lifted space to actions
        self.K = nn.Linear(lifting_dim, output_size)

        # Value function is quadratic in the lifted space
        self.V = PsdQuadratic(lifting_dim)

        # Linear dynamics matrix in the lifted space
        self.A = nn.Linear(lifting_dim, lifting_dim, bias=False)

        # Mapping from lifted space to observation space
        self.C = nn.Linear(lifting_dim, input_size, bias=False)

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)

    def forward_actor(self, y):
        z = self.phi(y)
        return self.K(z)

    def forward_critic(self, y):
        z = self.phi(y)
        return self.V(z)

    def forward_lifted_dynamics(self, z):
        return self.A(z)

    def predict_next_observation(self, y):
        z = self.phi(y)
        z_next = self.A(z)
        return self.C(z_next)

class KoopmanPolicy(ActorCriticPolicy):
    """
    A custom actor-critic policy that uses our custom Koopman architecture.
    Structured so that the SB3 PPO implementation can use our architecture
    directly.
    """
    def __init__(self, observation_space, action_space, lr_schedule,
                 lifting_dim, *args, **kwargs):
        self.lifting_dim = lifting_dim
        super().__init__(observation_space, action_space, lr_schedule, *args,
                **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = KoopmanMlpExtractor(
            self.features_dim, self.action_space.shape[0], 
            self.lifting_dim)
        
    def save(self, path):
        """
        Save the policy to disk, including some extra custom params. 
        """
        super().save(path, include=["lifting_dim"])

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
    def __init__(self, input_size, output_size, linear_system_type, num_blocks):
        super().__init__()

        # The custom network must have these output dimensions as attributes
        # with these names. The PPO implementation adds an additional linear
        # layer that maps from 'latent_dim_pi' to actions and from
        # 'latent_dim_vf' to values
        self.latent_dim_pi = output_size
        self.latent_dim_vf = 1

        # Policy
        assert linear_system_type in ["parallel", "series", "hierarchy"]
        assert num_blocks >= 1
        if linear_system_type == "parallel":
            self.policy_network = ParallelLinear(input_size, output_size, num_blocks)
        elif linear_system_type == "series":
            self.policy_network = SeriesLinear(input_size, output_size, num_blocks)
        elif linear_system_type == "hierarchy":
            self.policy_network = HierarchyLinear(input_size, output_size, num_blocks)

        # Value function
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
    def __init__(self, observation_space, action_space, lr_schedule, 
                 linear_system_type, num_blocks, *args, **kwargs):
        self.linear_system_type = linear_system_type
        self.num_blocks = num_blocks
        super().__init__(observation_space, action_space, lr_schedule, *args,
                **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = LinearMlpExtractor(
            self.features_dim, self.action_space.shape[0], 
            linear_system_type=self.linear_system_type,
            num_blocks=self.num_blocks)
        
    def save(self, path):
        """
        Save the policy to disk, including some extra custom params. 
        """
        super().save(path, include=["linear_system_type", "num_blocks"])
