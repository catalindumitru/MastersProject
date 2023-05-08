import torch
import torch.nn as nn
import numpy as np
from collections import deque
from network import CategoricalActorCriticNet
from storage import Storage
from utils import to_np, tensor, uniform_kernel, kernel_without_principal, random_sample
from environment import Environment


class RobustAgentPPO:
    def __init__(self, env: Environment):
        self.env = env

        # Lexicographic Robustness
        self.i = None
        self.mu = [0.0 for _ in range(1)]
        self.j = [0.0 for _ in range(1)]
        self.recent_losses = [deque(maxlen=25) for i in range(2)]
        self.beta = list(reversed(range(1, 3)))
        self.eta = [1e-3 * eta for eta in list(reversed(range(1, 3)))]
        self.second_order = False
        self.tau = 0.01
        self.rollout_length = 128
        self.gae_tau = 0.95
        self.entropy_weight = 0
        self.value_loss_weight = 0.1
        self.optimization_epochs = 10
        self.mini_batch_size = 32
        self.target_kl = 0.01
        self.ppo_ratio_clip = 0.2

        self.max_steps = 10000
        self.max_eval_steps = 1000
        self.episode_count = 100

        self.network = CategoricalActorCriticNet(
            env.state_count,
            env.action_count,
        )
        self.total_steps = 0
        self.state = self.env.sample_state()

        self.actor_opt = torch.optim.Adam(self.network.actor_params, 3e-4, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.network.actor_params, 1.5e-4, eps=1e-5)

        # anneal lr
        l = lambda f: 1 - f / self.max_steps
        self.lr_scheduler_policy = torch.optim.lr_scheduler.LambdaLR(
            self.actor_opt, lr_lambda=l
        )
        self.lr_scheduler_value = torch.optim.lr_scheduler.LambdaLR(
            self.critic_opt, lr_lambda=l
        )

        self.loss_bound = 0.5  # or 5?

    def reset(self):
        self.total_steps = 0
        self.state = self.env.sample_state()

    def step(self):
        storage = Storage(self.rollout_length)
        state = self.state
        theta = self.env.sample_theta(state)
        for _ in range(self.rollout_length):
            obs = torch.cat((state, theta), 0)
            prediction = self.network(obs)
            action = prediction["action"]

            next_state = self.env.take_action(state, action)
            next_theta = self.env.sample_theta(next_state)
            reward_A = self.env.R_A[state, action, theta]
            reward_P = self.env.R_P[state, action, theta]
            storage.feed(prediction)
            storage.feed({"state": tensor(state)})
            storage.feed({"theta": tensor(theta)})
            storage.feed(
                {
                    "reward_A": tensor(reward_A).unsqueeze(-1),
                    "reward_P": tensor(reward_P).unsqueeze(-1),
                }
            )
            state = next_state
            theta = next_theta
            self.total_steps += 1

        self.state = state
        storage.feed({"state": state})
        storage.feed({"theta": theta})
        obs = torch.cat((state, theta), 0)
        prediction = self.network(obs)
        storage.feed(prediction)
        storage.placeholder()

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
            [
                "log_pi_a",
                "action",
                "v",
                "ret",
                "advantage",
                "entropy",
                "pi",
                "state",
                "theta",
            ]
        )
        EntryCLS = entries.__class__
        entries = EntryCLS(*list(map(lambda x: x.detach(), entries)))
        entries.advantage.copy_(
            (entries.advantage - entries.advantage.mean()) / entries.advantage.std()
        )
        for _ in range(self.optimization_epochs):
            sampler = random_sample(
                np.arange(entries.state.size(0)), self.mini_batch_size
            )
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                entry = EntryCLS(*list(map(lambda x: x[batch_indices], entries)))
                obs = tensor(
                    [[state, theta] for state, theta in zip(entry.state, entry.theta)]
                )
                prediction = self.network(obs, entry.action)
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
                    if self.total_steps > self.max_steps / 3:
                        weights = self.compute_weights()
                        robust_loss = self.robust_loss(
                            entry.state, entry.theta, entry.pi
                        )
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

    def train(self):
        while self.total_steps < self.max_steps:
            self.step()

    def eval_step(self, obs):
        prediction = self.network(obs)
        action = to_np(prediction["action"]).squeeze()
        return action

    def eval_noisy_episode(self):
        self.reset()
        state = self.state
        total_reward_A = 0
        total_reward_P = 0
        discount_A = discount_P = 1
        while self.total_steps < self.max_eval_steps:
            theta = self.env.sample_theta(state)
            # theta_disturbed = uniform_kernel(self.env.theta_count)
            theta_disturbed = kernel_without_principal(state, self.env.mu)
            obs = torch.cat((state, theta_disturbed), 0)
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

        return np.mean(episodic_rewards_A), np.mean(episodic_rewards_P)

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
        # Update Lagrange parameters
        for i in range(1):
            self.mu[i] += self.eta[i] * (
                self.j[i] - self.tau * self.j[i] - self.recent_losses[i][-1]
            )
            self.mu[i] = max(self.mu[i], 0.0)
            # self.tau *= 0.999

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

    def robust_loss(self, states, thetas, actions):
        theta_disturbed = [
            tensor(kernel_without_principal(state, self.env.mu)) for state in states
        ]
        obs_disturbed = tensor(
            [[state, theta] for state, theta in zip(states, theta_disturbed)]
        )
        # actions = actions.view(
        #     actions.shape[0] // self.env.action_count, self.env.action_count
        # )
        target = self.network.actor(obs_disturbed)
        loss = 0.5 * (actions.detach() - target).pow(2).mean()
        return torch.clip(loss, -self.loss_bound, self.loss_bound)
