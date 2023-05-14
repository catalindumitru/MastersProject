import numpy as np
from environment import Environment
from torch.distributions import Categorical
from utils import tensor

from base_agent import BaseAgent


class BayesianAgent(BaseAgent):
    def __init__(self, env: Environment = None, principal_strategy=None):
        super().__init__(env, principal_strategy)

    def eval_episode(self, principal_strategy_disturbed):
        self.reset()
        state = self.state
        theta = self.theta
        total_reward = 0
        discount_A = 1
        while self.total_steps < self.max_eval_steps:
            signaled_action = Categorical(
                tensor(principal_strategy_disturbed[state, theta])
            ).sample()
            total_reward += discount_A * self.env.R_A[state, signaled_action, theta]

            state, theta = self.env.step(state, signaled_action)
            discount_A *= self.env.gamma_A
            self.total_steps += 1

        return total_reward
