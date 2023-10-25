import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

class MultiKoopmanPolicy(nn.Module):
    """
    A neural net that maps from a history of observations to control actions.
    Does so by building several linear maps, each of which is based on a learned
    Koopman linearization around a different fixed point of the system.
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Linear mapping from input (history of observations) to output (control)
        self.linear_system1 = nn.Linear(input_size, output_size, bias=False)

        # Second linear mapping from input to output, hopefully to capture a
        # Koopman linearization around a different equilibrium
        self.linear_system2 = nn.Linear(input_size, output_size, bias=False)

        # The chooser maps the input to a number between 0 and 1, which is used
        # to decide which linear system to use
        self.chooser = nn.Sequential(
            nn.Linear(input_size, 2, bias=False), 
            nn.Sigmoid()
        )

    def forward(self, observations):
        u1 = self.linear_system1(observations)
        u2 = self.linear_system2(observations)
        sigma = self.chooser(observations)
        return sigma[:,0:1]*u1 + sigma[:,1:2]*u2

class CustomNetwork(nn.Module):
    """
    A custom neural net for both the policy and the value function.
    """
    def __init__(self, input_size):
        super().__init__()

        # The custom network must have these output dimensions as attributes
        # with these names
        output_size = 1    # action space size for pendulum
        self.latent_dim_pi = output_size
        self.latent_dim_vf = 64

        # The PPO implementation adds an additional linear layer that maps from
        # 'latent_dim_pi' to actions
        self.policy_net = MultiKoopmanPolicy(input_size, output_size)
        self.value_net = nn.Sequential(
                nn.Linear(input_size, self.latent_dim_vf), nn.ReLU(),
                nn.Linear(self.latent_dim_vf, self.latent_dim_vf), nn.ReLU())

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)

    def forward_actor(self, x):
        return self.policy_net(x)

    def forward_critic(self, x):
        return self.value_net(x)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args,
            **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args,
                **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomNetwork(self.features_dim)

