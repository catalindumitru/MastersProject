from environment import Environment
from principal import Principal
from oblivious_robust_agent_a2c import ObliviousRobustAgentA2C
from oblivious_robust_agent_ppo import ObliviousRobustAgentPPO
from mindful_robust_agent_a2c import MindfulRobustAgentA2C
from mindful_robust_agent_ppo import MindfulRobustAgentPPO
from robust_agent_action import RobustAgentAction
from robust_agent_action_PPO import RobustAgentActionPPO
from robust_agent import RobustAgent
from robust_agent_PPO import RobustAgentPPO
from bayesian_agent import BayesianAgent
from bayesian_agent_action import BayesianAgentAction
from robust_agent_with_principal import RobustAgentWithPrincipal
from robust_agent_PPO_with_principal import RobustAgentPPOWithPrincipal
from utils import disturb_strategy
import numpy as np


class Game:
    keys = ["bayesian", "robust_A2C", "robust_A2C_with_principal"]

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

        agent = ObliviousRobustAgentA2C(env, principal_strategy_optimal)
        agent.train()
        results["oblivious"] = agent.eval_episodes(principal_strategy_disturbed)

        agent = ObliviousRobustAgentPPO(env, principal_strategy_optimal)
        agent.train()
        results["oblivious_PPO"] = agent.eval_episodes(principal_strategy_disturbed)

        agent = MindfulRobustAgentA2C(env, principal_strategy_optimal)
        agent.train()
        results["mindful"] = agent.eval_episodes(principal_strategy_disturbed)

        agent = MindfulRobustAgentPPO(env, principal_strategy_optimal)
        agent.train()
        results["mindful_PPO"] = agent.eval_episodes(principal_strategy_disturbed)

        # agent_bayesian = BayesianAgent(env, principal_strategy_optimal)
        # results["bayesian"] = agent_bayesian.eval_episodes(principal_strategy_disturbed)

        # agent = RobustAgentPPO(env)
        # agent.train()
        # results["robust_PPO"] = agent.eval_noisy_episodes()

        # agent = RobustAgentPPOWithPrincipal(env, principal_strategy_optimal)
        # agent.train()
        # results["robust_PPO_with_principal"] = agent.eval_noisy_episodes(
        #     principal_strategy_disturbed
        # )

        # agent = RobustAgent(env)
        # agent.train()
        # results["robust_A2C"] = agent.eval_noisy_episodes()

        # agent = RobustAgentWithPrincipal(env, principal_strategy_optimal)
        # agent.train()
        # results["robust_A2C_with_principal"] = agent.eval_noisy_episodes(
        #     principal_strategy_disturbed
        # )

        # agent = RobustAgentAction(env, principal_strategy_optimal)
        # agent.train()
        # results["robust_action"] = agent.eval_noisy_episodes()

        # agent = RobustAgentActionPPO(env, principal_strategy_optimal)
        # agent.train()
        # results["robust_action_PPO"] = agent.eval_noisy_episodes()

        # agent = BayesianAgentAction(env, principal_strategy_optimal)
        # results["bayesian"] = agent.eval_episodes()

        return results


if __name__ == "__main__":
    game = Game()
    print(game.play())
