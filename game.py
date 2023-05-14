from environment import Environment
from principal import Principal
from bayesian_agent import BayesianAgent
from oblivious_robust_agent_a2c import ObliviousRobustAgentA2C
from oblivious_robust_agent_ppo import ObliviousRobustAgentPPO
from mindful_robust_agent_a2c import MindfulRobustAgentA2C
from mindful_robust_agent_ppo import MindfulRobustAgentPPO
from utils import disturb_strategy
import numpy as np


class Game:
    keys = ["bayesian", "oblivious_A2C", "oblivious_PPO", "mindful_A2C", "mindful_PPO"]

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def play(self):
        results = {}
        env = Environment()

        principal = Principal(env)
        principal.train()

        principal_strategy_optimal = principal.get_optimal_strategy()
        principal_strategy_disturbed = disturb_strategy(
            env, principal_strategy_optimal, self.alpha
        )

        agent = BayesianAgent(env, principal_strategy_optimal)
        results["bayesian"] = agent.eval_episodes(principal_strategy_disturbed)

        agent = ObliviousRobustAgentA2C(env, principal_strategy_optimal)
        agent.train()
        results["oblivious_A2C"] = agent.eval_episodes(principal_strategy_disturbed)

        agent = ObliviousRobustAgentPPO(env, principal_strategy_optimal)
        agent.train()
        results["oblivious_PPO"] = agent.eval_episodes(principal_strategy_disturbed)

        agent = MindfulRobustAgentA2C(env, principal_strategy_optimal)
        agent.train()
        results["mindful_A2C"] = agent.eval_episodes(principal_strategy_disturbed)

        agent = MindfulRobustAgentPPO(env, principal_strategy_optimal)
        agent.train()
        results["mindful_PPO"] = agent.eval_episodes(principal_strategy_disturbed)

        return results


if __name__ == "__main__":
    game = Game()
    print(game.play())
