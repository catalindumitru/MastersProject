import torch
import torch.nn as nn
import numpy as np
from collections import deque
from network import CategoricalActorCriticNet
from storage import Storage
from utils import (
    to_np,
    tensor,
    uniform_kernel,
    kernel_without_principal,
    sample_signal,
    disturb_signal,
)
from environment import Environment


class RobustAgentAction:
    def __init__(self, env: Environment, principal_policy):
        self.env = env
        self.principal_policy = principal_policy

        # Lexicographic Robustness
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
        self.entropy_weight = 0.01  # 0.01
        self.value_loss_weight = 1
        self.gradient_clip = 0.5

        self.max_steps = 10000
        self.max_eval_steps = 1000
        self.episode_count = 100

        self.network = CategoricalActorCriticNet(
            env.state_count,
            env.action_count,
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001, eps=1e-8)
        self.total_steps = 0
        self.state = tensor(self.env.S[0]).to(torch.int)

    def reset(self):
        self.total_steps = 0
        self.state = tensor(self.env.S[0]).to(torch.int)

    def step(self):
        storage = Storage(self.rollout_length)
        state = self.state
        theta = self.env.sample_theta(state)
        signal = sample_signal(self.principal_policy, state, theta)
        for _ in range(self.rollout_length):
            obs = torch.stack((state, signal))
            prediction = self.network(obs)
            action = prediction["action"]
            # print("ACtion", action)
            next_state = self.env.take_action(state, action)
            next_theta = self.env.sample_theta(next_state)
            next_signal = sample_signal(self.principal_policy, next_state, next_theta)
            reward_A = self.env.R_A[state, action, theta]
            reward_P = self.env.R_P[state, action, theta]
            # print(self.env.R_A[state, :, theta])

            storage.feed(prediction)
            storage.feed({"state": state})
            storage.feed({"signal": signal})
            storage.feed(
                {
                    "reward_A": tensor(reward_A).unsqueeze(-1),
                    "reward_P": tensor(reward_P).unsqueeze(-1),
                }
            )
            state = next_state
            theta = next_theta
            signal = next_signal
            self.total_steps += 1

        self.state = state
        storage.feed({"state": state})
        storage.feed({"signal": signal})
        obs = torch.stack((state, signal))
        prediction = self.network(obs)
        storage.feed(prediction)
        storage.placeholder()
        # print(storage.extract(["pi"])[0].size())

        advantages = tensor(np.zeros((1, 1)))
        returns = prediction["v"].detach()
        for i in reversed(range(self.rollout_length)):
            returns = storage.reward_A[i] + self.env.gamma_A * returns
            td_error = (
                storage.reward_A[i] + self.env.gamma_A * storage.v[i + 1] - storage.v[i]
            )
            advantages = advantages * self.gae_tau * self.env.gamma_A + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        entries = storage.extract(
            ["log_pi_a", "v", "ret", "advantage", "entropy", "pi", "state", "signal"]
        )
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()

        weights = self.compute_weights()
        if self.total_steps > self.max_steps / 3:
            robust_loss = self.robust_loss(entries.state, entries.signal, entries.pi)
            self.update_lagrange()
            # print(robust_loss)
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
        nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
        self.optimizer.step()
        # print("Actual Reward", storage.reward_A)

    def train(self):
        while self.total_steps < self.max_steps:
            self.step()

    def eval_step(self, obs):
        prediction = self.network(obs)
        action = to_np(prediction["action"])
        return action

    def eval_noisy_episode(self):
        self.reset()
        state = self.state
        total_reward_A = 0
        total_reward_P = 0
        discount_A = discount_P = 1
        while self.total_steps < self.max_eval_steps:
            # print(self.total_steps)
            theta = self.env.sample_theta(state)
            signal = sample_signal(self.principal_policy, state, theta)
            disturbed_signal = tensor(disturb_signal(signal, 0.66))
            obs = torch.stack((state, disturbed_signal))
            action = self.eval_step(obs)
            total_reward_A += discount_A * self.env.R_A[state, action, theta]
            total_reward_P += discount_P * self.env.R_P[state, action, theta]

            state = self.env.take_action(state, action)
            discount_A *= self.env.gamma_A
            discount_P *= self.env.gamma_P
            self.total_steps += 1

        return total_reward_A, total_reward_P

    def eval_noisy_episodes(self):
        episodic_rewards_A = []
        episodic_rewards_P = []
        for ep in range(self.episode_count):
            # print("Ep: ", ep)
            total_rewards_A, total_rewards_P = self.eval_noisy_episode()
            episodic_rewards_A.append(np.sum(total_rewards_A))
            episodic_rewards_P.append(np.sum(total_rewards_P))

        return np.mean(episodic_rewards_A)  # , np.mean(episodic_rewards_P)

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

    def robust_loss(self, states, signals, actions):
        signal_disturbed = [tensor(disturb_signal(signal, 0.66)) for signal in signals]
        obs_disturbed = torch.stack(
            [
                torch.stack((state, signal))
                for state, signal in zip(states, signal_disturbed)
            ]
        )
        # print(states, thetas, actions)
        # actions = actions.view(
        #     actions.shape[0] // self.env.action_count, self.env.action_count
        # )
        target = self.network.actor(obs_disturbed)
        assert len(actions) == len(target)
        loss = 0.5 * (actions.detach() - target).pow(2).mean()
        return torch.clip(loss, -0.2, 0.2)
