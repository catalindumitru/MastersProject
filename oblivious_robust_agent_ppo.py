import torch
import numpy as np
from torch.distributions import Categorical

from base_robust_agent import BaseRobustAgent
from base_robust_agent_ppo import BaseRobustAgentPPO
from utils import tensor
from environment import Environment


class ObliviousRobustAgentPPO(BaseRobustAgent, BaseRobustAgentPPO):
    def __init__(self, env: Environment = None, principal_strategy=None):
        BaseRobustAgent.__init__(self, env, principal_strategy)
        BaseRobustAgentPPO.__init__(self, 3, env.action_count, self.max_train_steps)
        self.reset_meta_state(principal_strategy)

    def train_kernel(self, meta_states):
        disturbed_meta_states = []
        for state, signal, _ in meta_states:
            disturbed_meta_states.append(
                torch.stack(
                    (state, signal, Categorical(tensor(self.env.mu[state])).sample())
                )
            )
        return torch.stack(disturbed_meta_states)
