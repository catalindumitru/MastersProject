import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from utils import tensor
from torch.distributions import Categorical
from utils import layer_init


class CategoricalActorCriticNet(nn.Module):
    def __init__(
        self,
        state_count,
        action_count,
    ):
        super(CategoricalActorCriticNet, self).__init__()

        self.phi_body = FCBody(state_count)
        self.actor_body = DummyBody(self.phi_body.feature_dim)
        self.critic_body = DummyBody(self.phi_body.feature_dim)

        self.fc_action = layer_init(
            nn.Linear(self.actor_body.feature_dim, action_count), 1e-3
        )
        self.fc_critic = layer_init(nn.Linear(self.critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(
            self.fc_action.parameters()
        )
        self.critic_params = list(self.critic_body.parameters()) + list(
            self.fc_critic.parameters()
        )
        self.phi_params = list(self.phi_body.parameters())

        # self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        logits = self.fc_action(phi_a)
        v = self.fc_critic(phi_v)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample().unsqueeze(0)
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {
            "action": action,
            "log_pi_a": log_prob,
            "entropy": entropy,
            "v": v,
            "pi": F.softmax(logits, dim=-1),
        }  # dist.probs}

    def actor(self, obs):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        logits = self.fc_action(phi_a)
        return F.softmax(logits, dim=-1)


class DummyBody(nn.Module):
    def __init__(self, state_count):
        super(DummyBody, self).__init__()
        self.feature_dim = state_count

    def forward(self, x):
        return x


class FCBody(nn.Module):
    def __init__(
        self, state_dim, hidden_units=(64, 64), gate=F.relu, noisy_linear=False
    ):
        super(FCBody, self).__init__()
        dims = (2,) + hidden_units
        if noisy_linear:
            self.layers = nn.ModuleList(
                [
                    NoisyLinear(dim_in, dim_out)
                    for dim_in, dim_out in zip(dims[:-1], dims[1:])
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    layer_init(nn.Linear(dim_in, dim_out))
                    for dim_in, dim_out in zip(dims[:-1], dims[1:])
                ]
            )

        self.gate = gate
        self.feature_dim = dims[-1]
        self.noisy_linear = noisy_linear

    def reset_noise(self):
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()

    def forward(self, x):
        x = x.to(torch.float32)
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
            torch.zeros((out_features, in_features)), requires_grad=True
        )
        self.weight_sigma = nn.Parameter(
            torch.zeros((out_features, in_features)), requires_grad=True
        )
        self.register_buffer("weight_epsilon", torch.zeros((out_features, in_features)))

        self.bias_mu = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.bias_sigma = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.register_buffer("noise_in", torch.zeros(in_features))
        self.register_buffer("noise_out_weight", torch.zeros(out_features))
        self.register_buffer("noise_out_bias", torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.weight_sigma.size(1))
        )

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        self.noise_in.normal_(std=0.1)
        self.noise_out_weight.normal_(std=0.1)
        self.noise_out_bias.normal_(std=0.1)

        self.weight_epsilon.copy_(
            self.transform_noise(self.noise_out_weight).ger(
                self.transform_noise(self.noise_in)
            )
        )
        self.bias_epsilon.copy_(self.transform_noise(self.noise_out_bias))

    def transform_noise(self, x):
        return x.sign().mul(x.abs().sqrt())
