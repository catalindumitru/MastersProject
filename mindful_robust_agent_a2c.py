import torch
import numpy as np
from torch.distributions import Categorical

from base_robust_agent import BaseRobustAgent
from base_robust_agent_a2c import BaseRobustAgentA2C
from utils import tensor
from environment import Environment


class MindfulRobustAgentA2C(BaseRobustAgent, BaseRobustAgentA2C):
    def __init__(self, env: Environment = None, principal_strategy=None):
        BaseRobustAgent.__init__(self, env, principal_strategy)
        BaseRobustAgentA2C.__init__(self, 3, env.action_count)
        self.reset_meta_state(principal_strategy)

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

    def train_kernel(self, meta_states):
        disturbed_meta_states = []
        for state, signal, _ in meta_states:
            disturbed_meta_states.append(
                torch.stack(
                    (state, signal, Categorical(tensor(self.env.mu[state])).sample())
                )
            )
        return torch.stack(disturbed_meta_states)

    def deployment_kernel(self, meta_state, principal_strategy_disturbed):
        state, signal, _ = meta_state

        kernel = np.zeros((self.env.theta_count))
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
