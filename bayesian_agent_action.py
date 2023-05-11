import numpy as np
import gurobipy as gp
from gurobipy import GRB
from environment import Environment
from torch.distributions import Categorical
from utils import tensor, sample_signal, disturb_signal


class BayesianAgentAction:
    def __init__(self, env: Environment, principal_policy) -> None:
        self.env = env

        self.max_steps = 1000
        self.episode_count = 100

        self.total_steps = 0
        self.state = self.env.S[0]
        self.principal_policy = principal_policy

    def reset(self):
        self.total_steps = 0
        self.state = self.env.S[0]

    def eval_episode(self):
        self.reset()
        state = self.state
        total_reward_A = 0
        total_reward_P = 0
        discount_A = discount_P = 1
        while self.total_steps < self.max_steps:
            # print(self.total_steps)
            theta = self.env.sample_theta(state)
            signal = sample_signal(self.principal_policy, state, theta)
            disturbed_signal = tensor(disturb_signal(signal, 0.66))
            total_reward_A += discount_A * self.env.R_A[state, disturbed_signal, theta]
            total_reward_P += discount_P * self.env.R_P[state, disturbed_signal, theta]

            state = self.env.take_action(state, disturbed_signal)
            discount_A *= self.env.gamma_A
            discount_P *= self.env.gamma_P
            self.total_steps += 1

        return total_reward_A, total_reward_P

    def eval_episodes(self):
        episodic_rewards_A = []
        episodic_rewards_P = []
        for ep in range(self.episode_count):
            # print("Ep: ", ep)
            total_rewards_A, total_rewards_P = self.eval_episode()
            episodic_rewards_A.append(np.sum(total_rewards_A))
            episodic_rewards_P.append(np.sum(total_rewards_P))

        return np.mean(episodic_rewards_A)  # , np.mean(episodic_rewards_P)
