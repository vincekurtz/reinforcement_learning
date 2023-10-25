import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

class KoopmanMlpExtractor(nn.Module):
    """
    A custom neural net for both the policy and the value function.
    """
    def __init__(self, input_size, output_size):
        super().__init__()

        # The custom network must have these output dimensions as attributes
        # with these names. The PPO implementation adds an additional linear
        # layer that maps from 'latent_dim_pi' to actions and from
        # 'latent_dim_vf' to values
        self.latent_dim_pi = output_size
        self.latent_dim_vf = 1

        # Policy is piecewise linear with a learned switching surface
        self.num_linear_systems = 3
        
        self.linear_systems = nn.ModuleList([
            nn.Linear(input_size, output_size, bias=False)
            for _ in range(self.num_linear_systems)])

        # We define a switching surface between regimes with a neural net. This
        # network outputs scores in [0,1] that determine which linear layer is active
        self.chooser = nn.Sequential(
            nn.Linear(input_size, self.num_linear_systems, bias=False), 
            nn.Sigmoid())
        
        # Value function is a simple MLP
        self.value_net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, self.latent_dim_vf))

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
        output /= sigma.sum(dim=1, keepdim=True)

        return output

    def forward_critic(self, x):
        return self.value_net(x)

class KoopmanPolicy(ActorCriticPolicy):
    """
    A custom actor-critic policy that uses our custom Koopman architecture.
    Structured so that the SB3 PPO implementation can use our architecture
    directly.
    """
    def __init__(self, observation_space, action_space, lr_schedule, *args,
            **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args,
                **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = KoopmanMlpExtractor(
            self.features_dim, self.action_space.shape[0])
