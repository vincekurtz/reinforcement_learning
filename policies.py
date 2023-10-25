import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

class Quadratic(nn.Module):
    """
    A simple quadratic function

        y = x'Ax + b'x + c

    where A, b, and c are learned parameters.
    """
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.A = nn.Parameter(torch.randn(input_size, input_size), requires_grad=True)
        self.b = nn.Parameter(torch.randn(input_size), requires_grad=True)
        self.c = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        y = (x @ self.A @ x.T).diag() + x @ self.b + self.c
        return y.view(-1, 1)

class KoopmanNetwork(nn.Module):
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
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Sigmoid()
        )
        #self.chooser = nn.Sequential(
        #    nn.Linear(input_size, 2, bias=False), 
        #    nn.Sigmoid()
        #)

    def forward(self, observations):
        u1 = self.linear_system1(observations)
        u2 = self.linear_system2(observations)
        sigma = self.chooser(observations)
        return sigma[:,0:1]*u1 + sigma[:,1:2]*u2
    
class PiecewiseQuadratic(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = 1

        self.quadratic1 = Quadratic(input_size)
        self.quadratic2 = Quadratic(input_size)

        self.chooser = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Sigmoid()
        )

    def forward(self, x):
        y1 = self.quadratic1(x)
        y2 = self.quadratic2(x)
        sigma = self.chooser(x)
        return sigma[:,0:1]*y1 + sigma[:,1:2]*y2

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

        # Policy is piecewise linear
        self.linear_system1 = nn.Linear(input_size, output_size, bias=False)
        self.linear_system2 = nn.Linear(input_size, output_size, bias=False)

        # Value function is piecewise quadratic
        self.quadratic1 = Quadratic(input_size)
        self.quadratic2 = Quadratic(input_size)

        # We define a switching surface between regimes with a neural net. This
        # network outputs a number between 0 and 1
        self.chooser = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Sigmoid())

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)

    def forward_actor(self, x):
        y1 = self.linear_system1(x)
        y2 = self.linear_system2(x)
        sigma = self.chooser(x)
        return sigma[:,0:1]*y1 + sigma[:,1:2]*y2

    def forward_critic(self, x):
        y1 = self.quadratic1(x)
        y2 = self.quadratic2(x)
        sigma = self.chooser(x)
        return sigma[:,0:1]*y1 + sigma[:,1:2]*y2

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
