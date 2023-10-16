import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomNetwork(nn.Module):
    """
    A custom neural net for both the policy and the value function.
    """
    def __init__(self, input_size):
        super().__init__()

        # The custom network must have these output dimensions as attributes
        # with these names
        self.latent_dim_pi = 64
        self.latent_dim_vf = 64

        #self.policy_net = nn.Sequential(
        #        nn.Linear(input_size, self.latent_dim_pi), nn.ReLU(),
        #        nn.Linear(self.latent_dim_pi, self.latent_dim_pi), nn.ReLU())
        self.policy_net = nn.Linear(input_size, self.latent_dim_pi)
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

