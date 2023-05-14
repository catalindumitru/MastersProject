import torch
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
from numpy import arange, zeros

from environment import Environment
from base_agent import BaseAgent
from base_robust_agent_a2c import BaseRobustAgentA2C
from base_robust_agent_ppo import BaseRobustAgentPPO
from utils import to_np, tensor, random_sample
from storage import Storage


class BaseRobustAgent(BaseAgent):
    def __init__(self, env: Environment = None, principal_strategy=None):
        super().__init__(env, principal_strategy)
        self.max_train_steps = 1000
        
    def reset_meta_state(self, principal_strategy):
        self.reset()
        state = self.state
        theta = self.theta
        signal = Categorical(tensor(principal_strategy[state, theta])).sample()
        self.meta_state = torch.stack((state, signal, theta))

    def next_meta_state(self, action, principal_strategy):
        state, signal, theta = self.meta_state
        next_state = self.env.take_action(state, action)
        next_theta = self.env.sample_theta(next_state)
        next_signal = Categorical(
            tensor(principal_strategy[next_state, next_theta])
        ).sample()

        self.state = next_state
        self.theta = next_theta
        self.meta_state = torch.stack((next_state, next_signal, next_theta))

    def train(self):
        while self.total_steps < self.max_train_steps:
            self.train_step()

    def train_step(self):
        storage = self.rollout_phase()

        if isinstance(self, BaseRobustAgentA2C):
            self.a2c_optimise(storage)
        elif isinstance(self, BaseRobustAgentPPO):
            self.ppo_optimise(storage)

    def rollout_phase(self):
        storage = Storage(self.rollout_length)
        for _ in range(self.rollout_length):
            prediction = self.network(self.meta_state)
            action = to_np(prediction["action"])
            reward_A = self.env.R_A[self.state, action, self.theta]

            storage.feed(prediction)
            storage.feed({"meta_state": self.meta_state})

            storage.feed(
                {
                    "reward_A": tensor(reward_A).unsqueeze(-1),
                }
            )

            self.next_meta_state(action, self.principal_strategy)
            self.total_steps += 1

        storage.feed({"meta_state": self.meta_state})
        prediction = self.network(self.meta_state)
        storage.feed(prediction)
        storage.placeholder()

        advantages = tensor(zeros((1, 1)))
        returns = prediction["v"].detach()
        for i in reversed(range(self.rollout_length)):
            returns = storage.reward_A[i] + self.env.gamma_A * returns
            td_error = (
                storage.reward_A[i] + self.env.gamma_A * storage.v[i + 1] - storage.v[i]
            )
            advantages = advantages * self.gae_tau * self.env.gamma_A + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        return storage

    def a2c_optimise(self, storage):
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

    def ppo_optimise(self, storage):
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
                    -torch.min(obj, obj_clipped).mean()
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

    def converged(self, tolerance=0.1, bound=0.01, minimum_updates=5):
        # If not enough updates have been performed, assume not converged
        if len(self.recent_losses[self.i]) < minimum_updates:
            return False
        else:
            l_mean = torch.tensor(self.recent_losses[self.i]).mean().float()
            # If the mean loss is lower than some absolute bound, assume converged
            if l_mean < bound:
                return True
            # Else check if the max of the recent losses are sufficiently close to the mean, if so then assume converged
            else:
                l_max = max(self.recent_losses[self.i]).float()
                if l_max > (1.0 + tolerance) * l_mean:
                    return False

        return True

    def update_lagrange(self):
        # Save relevant loss information for updating Lagrange parameters
        for i in range(1):
            self.j[i] = torch.tensor(self.recent_losses[i]).mean()
        # Update Lagrange parameters, mu==lambda
        for i in range(1):
            self.mu[i] += self.eta[i] * (
                self.j[i] - self.tau * self.j[i] - self.recent_losses[i][-1]
            )
            self.mu[i] = max(self.mu[i], 0.0)

    def compute_weights(self):
        reward_range = 2
        # Compute weights for different components of the first order objective terms
        first_order = []
        for i in range(reward_range - 1):
            w = self.beta[i] + self.mu[i] * sum(
                [self.beta[j] for j in range(i + 1, reward_range)]
            )
            first_order.append(w)
        first_order.append(self.beta[reward_range - 1])
        first_order_weights = torch.tensor(first_order)
        return first_order_weights

    def robust_loss(self, disturbed_meta_states, actions):
        target = self.network.actor(disturbed_meta_states)
        loss = 0.5 * (actions.detach() - target).pow(2).mean()
        return torch.clip(loss, -self.loss_bound, self.loss_bound)

    def kernel_T_alpha(self, meta_state, principal_strategy_disturbed):
        state, signal, _ = meta_state

        kernel = zeros((self.env.theta_count))
        denominator = 0
        for t in self.env.Theta:
            denominator += (
                principal_strategy_disturbed[state, t, signal] * self.env.mu[state, t]
            )
        for t in self.env.Theta:
            kernel[t] = (
                principal_strategy_disturbed[state, t, signal] * self.env.mu[state, t]
            ) / denominator

        return torch.stack((state, signal, Categorical(tensor(kernel)).sample()))

    def deployment_kernel(self, meta_state, principal_strategy_disturbed):
        return self.kernel_T_alpha(meta_state, principal_strategy_disturbed)

    def eval_episode(self, principal_strategy_disturbed=None):
        self.reset_meta_state(principal_strategy_disturbed)

        total_reward = 0
        discount_A = 1
        while self.total_steps < self.max_eval_steps:
            disturbed_meta_state = self.deployment_kernel(
                self.meta_state, principal_strategy_disturbed
            )
            prediction = self.network(disturbed_meta_state)
            action = to_np(prediction["action"])
            total_reward += discount_A * self.env.R_A[self.state, action, self.theta]

            self.next_meta_state(action, principal_strategy_disturbed)
            discount_A *= self.env.gamma_A
            self.total_steps += 1

        return total_reward
