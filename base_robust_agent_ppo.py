from torch import min as torch_min
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from numpy import arange
from collections import deque

from network import CategoricalActorCriticNet
from utils import random_sample, tensor


class BaseRobustAgentPPO:
    def __init__(self, meta_state_dim, action_count, max_train_steps):
        self.rollout_length = 32
        self.gae_tau = 0.95
        self.entropy_weight = 0.01
        self.value_loss_weight = 0.1
        self.optimization_epochs = 10
        self.mini_batch_size = 8
        self.target_kl = 0.01
        self.ppo_ratio_clip = 0.2

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

    def optimise(self, storage):
        entries = storage.extract(
            [
                "log_pi_a",
                "action",
                "v",
                "ret",
                "advantage",
                "entropy",
                "pi",
                "meta_state",
            ]
        )
        EntryCLS = entries.__class__
        entries = EntryCLS(*list(map(lambda x: x.detach(), entries)))
        entries.advantage.copy_(
            (entries.advantage - entries.advantage.mean()) / entries.advantage.std()
        )

        for _ in range(self.optimization_epochs):
            sampler = random_sample(
                arange(entries.meta_state.size(0)), self.mini_batch_size
            )
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                entry = EntryCLS(*list(map(lambda x: x[batch_indices], entries)))

                prediction = self.network(entry.meta_state, entry.action)
                ratio = (prediction["log_pi_a"] - entry.log_pi_a).exp()
                obj = ratio * entry.advantage
                obj_clipped = (
                    ratio.clamp(
                        1.0 - self.ppo_ratio_clip,
                        1.0 + self.ppo_ratio_clip,
                    )
                    * entry.advantage
                )
                policy_loss = (
                    -torch_min(obj, obj_clipped).mean()
                    - self.entropy_weight * prediction["entropy"].mean()
                )

                value_loss = 0.5 * (entry.ret - prediction["v"]).pow(2).mean()
                approx_kl = (entry.log_pi_a - prediction["log_pi_a"]).mean()

                if approx_kl <= 1.5 * self.target_kl:
                    if self.total_steps > 2 * self.max_train_steps / 3:
                        weights = self.compute_weights()
                        disturbed_meta_states = self.train_kernel(entry.meta_state)
                        robust_loss = self.robust_loss(disturbed_meta_states, entry.pi)
                        self.recent_losses[0].append(-policy_loss.mean())
                        self.recent_losses[1].append(robust_loss)
                        self.actor_opt.zero_grad()
                        (policy_loss * weights[0] + robust_loss * weights[1]).backward(
                            retain_graph=True
                        )
                        self.actor_opt.step()
                    else:
                        self.actor_opt.zero_grad()
                        policy_loss.backward(retain_graph=True)
                        self.actor_opt.step()
                self.critic_opt.zero_grad()
                value_loss.backward(retain_graph=True)
                self.critic_opt.step()

        self.lr_scheduler_policy.step()
        self.lr_scheduler_value.step()
