import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import tensor
from torch.distributions import Categorical
from utils import layer_init


class CategoricalActorCriticNet(nn.Module):
    def __init__(
        self,
        state_dim,
        action_count,
    ):
        super(CategoricalActorCriticNet, self).__init__()

        self.phi_body = FCBody(state_dim)
        self.actor_body = DummyBody(self.phi_body.feature_dim)
        self.critic_body = DummyBody(self.phi_body.feature_dim)

        # self.phi_body = DummyBody(state_count)
        # self.actor_body = FCBody(self.phi_body.feature_dim)
        # self.critic_body = FCBody(self.phi_body.feature_dim)

        # self.phi_body = DummyBody(2)
        # self.actor_body = DummyBody(2)
        # self.critic_body = DummyBody(2)

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

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        logits = self.fc_action(phi_a)
        v = self.fc_critic(phi_v)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {
            "action": action,
            "log_pi_a": log_prob,
            "entropy": entropy,
            "v": v,
            "pi": F.softmax(logits, dim=-1),
        }

    def actor(self, obs):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        logits = self.fc_action(phi_a)
        return F.softmax(logits, dim=-1)


class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x.to(torch.float32)


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units

        self.layers = nn.ModuleList(
            [
                layer_init(nn.Linear(dim_in, dim_out))
                for dim_in, dim_out in zip(dims[:-1], dims[1:])
            ]
        )

        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        x = x.to(torch.float32)
        for layer in self.layers:
            x = self.gate(layer(x))
        return x
