from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from collections import deque

from network import CategoricalActorCriticNet


class BaseRobustAgentPPO:
    def __init__(self, meta_state_dim, action_count, max_train_steps):
        # LR hyperparameters

        self.i = None
        self.mu = [0.0 for _ in range(1)]
        self.j = [0.0 for _ in range(1)]
        self.recent_losses = [deque(maxlen=25) for i in range(2)]
        self.beta = list(reversed(range(1, 3)))
        self.eta = [1e-3 * eta for eta in list(reversed(range(1, 3)))]
        self.second_order = False
        self.tau = 0.01
        self.rollout_length = 32
        self.gae_tau = 0.95
        self.entropy_weight = 0.01
        self.value_loss_weight = 0.1
        self.optimization_epochs = 10
        self.mini_batch_size = 8
        self.target_kl = 0.01
        self.ppo_ratio_clip = 0.2
        self.loss_bound = 5

        self.network = CategoricalActorCriticNet(
            state_dim=meta_state_dim,
            action_count=action_count,
        )
        self.optimizer = Adam(self.network.parameters(), lr=0.001, eps=1e-8)

        self.actor_opt = Adam(self.network.actor_params, 3e-4, eps=1e-5)
        self.critic_opt = Adam(self.network.actor_params, 1.5e-4, eps=1e-5)

        # anneal lr
        l = lambda f: 1 - f / max_train_steps
        self.lr_scheduler_policy = LambdaLR(self.actor_opt, lr_lambda=l)
        self.lr_scheduler_value = LambdaLR(self.critic_opt, lr_lambda=l)
