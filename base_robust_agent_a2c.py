from collections import deque
from torch.optim import Adam

from network import CategoricalActorCriticNet


class BaseRobustAgentA2C:
    def __init__(self, meta_state_dim, action_count):
        # LR hyperparameters

        self.i = None
        self.mu = [0.0 for _ in range(1)]
        self.j = [0.0 for _ in range(1)]
        self.recent_losses = [deque(maxlen=25) for i in range(2)]
        self.beta = list(reversed(range(1, 3)))
        self.eta = [1e-3 * eta for eta in list(reversed(range(1, 3)))]
        self.tau = 0.01
        self.rollout_length = 32
        self.gae_tau = 0.95
        self.entropy_weight = 0.01
        self.value_loss_weight = 1
        self.gradient_clip = 0.5
        self.loss_bound = 0.2

        self.network = CategoricalActorCriticNet(
            state_dim=meta_state_dim,
            action_count=action_count,
        )
        self.optimizer = Adam(self.network.parameters(), lr=0.001, eps=1e-8)
