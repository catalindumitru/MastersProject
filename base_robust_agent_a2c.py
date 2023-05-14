from collections import deque
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from network import CategoricalActorCriticNet
from utils import tensor

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

    def optimise(self, storage):
        entries = storage.extract(
            [
                "log_pi_a",
                "v",
                "ret",
                "advantage",
                "entropy",
                "pi",
                "meta_state",
            ]
        )
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()

        weights = self.compute_weights()
        if self.total_steps > 2 * self.max_train_steps / 3:
            disturbed_meta_states = self.train_kernel(entries.meta_state)
            robust_loss = self.robust_loss(disturbed_meta_states, entries.pi)
            self.update_lagrange()
        else:
            robust_loss = tensor(0)
        self.recent_losses[0].append(-policy_loss)
        self.recent_losses[1].append(-robust_loss)
        self.optimizer.zero_grad()
        (
            policy_loss
            + robust_loss * weights[1] / weights[0]
            - self.entropy_weight * entropy_loss
            + self.value_loss_weight * value_loss
        ).backward()
        clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.optimizer.step()
        