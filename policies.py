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
