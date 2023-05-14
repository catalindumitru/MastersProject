import torch
from torch.distributions import Categorical
from numpy import zeros
from collections import deque

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

        # LR Hyperparameters
        self.i = None
        self.mu = [0.0 for _ in range(1)]
        self.j = [0.0 for _ in range(1)]
        self.recent_losses = [deque(maxlen=25) for i in range(2)]
        self.beta = list(reversed(range(1, 3)))
        self.eta = [1e-3 * eta for eta in list(reversed(range(1, 3)))]
        self.tau = 0.01
        self.loss_bound = 0.2

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
        self.optimise(storage)

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
