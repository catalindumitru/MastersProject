import torch

from base_robust_agent import BaseRobustAgent
from base_robust_agent_a2c import BaseRobustAgentA2C
from utils import disturb_strategy
from environment import Environment


class MindfulRobustAgentA2C(BaseRobustAgent, BaseRobustAgentA2C):
    def __init__(self, env: Environment = None, principal_strategy=None):
        BaseRobustAgent.__init__(self, env, principal_strategy)
        BaseRobustAgentA2C.__init__(self, 3, env.action_count)
        self.principal_strategy_train = disturb_strategy(env, principal_strategy, 0.4)
        self.reset_meta_state(principal_strategy)

    def train_kernel(self, meta_states):
        disturbed_meta_states = []
        for meta_state in meta_states:
            disturbed_meta_states.append(
                self.kernel_T_alpha(meta_state, self.principal_strategy_train)
            )
        return torch.stack(disturbed_meta_states)
