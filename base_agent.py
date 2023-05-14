from environment import Environment
from numpy import mean


class BaseAgent:
    def __init__(self, env: Environment = None, principal_strategy=None):
        self.env = env
        self.principal_strategy = principal_strategy

        self.max_eval_steps = 1000
        self.eval_episode_count = 100

        self.reset()

    def reset(self):
        self.total_steps = 0
        self.state = self.env.init_state
        self.theta = self.env.sample_theta(self.state)

    def eval_episodes(self, principal_strategy_disturbed=None):
        episodic_rewards = []
        for _ in range(self.eval_episode_count):
            episodic_rewards.append(self.eval_episode(principal_strategy_disturbed))

        return mean(episodic_rewards)

    def eval_episode(self, principal_strategy_disturbed):
        raise Exception("Method not implemented")
