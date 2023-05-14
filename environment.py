import numpy as np
import torch
from utils import random_distribution
from torch.distributions import Categorical
from torch import tensor


class Environment:
    def __init__(
        self,
        state_count=50,
        action_count=25,
        theta_count=25,
        gamma_A=0.8,
        gamma_P=0.8,
        terminal_states=0,
        lb_P=0,
        ub_P=1,
        lb_A=0,
        ub_A=1,
    ):
        self.state_count = state_count
        self.action_count = action_count
        self.theta_count = theta_count
        self.S = range(state_count)
        self.U = range(action_count)
        self.Theta = range(theta_count)

        self.init_distribution = random_distribution(state_count)
        self.init_state = Categorical(tensor(self.init_distribution)).sample()
        self.gamma_A = gamma_A
        self.gamma_P = gamma_P

        self.mu = np.ones((state_count, theta_count))
        self.P = np.ones((state_count, action_count, state_count))
        for s in range(state_count):
            self.mu[s, :] = random_distribution(theta_count)
            for a in range(action_count):
                self.P[s, a, :] = random_distribution(state_count)

        self.R_P = np.random.uniform(
            low=lb_P, high=ub_P, size=(state_count, action_count, theta_count)
        )
        self.R_A = np.random.uniform(
            low=lb_A, high=ub_A, size=(state_count, action_count, theta_count)
        )

        for t in range(state_count - terminal_states, state_count):  # maybe -1?
            for a in range(action_count):
                self.P[t, a, :] = 0
                self.P[t, a, t] = 1
            self.R_P[t, :, :] = 0
            self.R_A[t, :, :] = 0

    def take_action(self, state, action):
        return Categorical(tensor(self.P[state, action, :])).sample()

    def sample_theta(self, state):
        return Categorical(tensor(self.mu[state, :])).sample()

    def step(self, state, action):
        next_state = self.take_action(state, action)
        next_theta = self.sample_theta(next_state)
        return next_state, next_theta
