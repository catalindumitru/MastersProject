from environment import Environment
from principal import Principal
from oblivious_robust_agent_a2c import ObliviousRobustAgentA2C
from oblivious_robust_agent_ppo import ObliviousRobustAgentPPO
from robust_agent_action import RobustAgentAction
from robust_agent_action_PPO import RobustAgentActionPPO
from robust_agent import RobustAgent
from robust_agent_PPO import RobustAgentPPO
from bayesian_agent import BayesianAgent
from bayesian_agent_action import BayesianAgentAction
from robust_agent_with_principal import RobustAgentWithPrincipal
from robust_agent_PPO_with_principal import RobustAgentPPOWithPrincipal
from utils import noisy_distribution
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

        principal_policy_optimal = principal.get_optimal_strategy()
        principal_policy_noisy = np.zeros(
            (env.state_count, env.theta_count, env.action_count)
        )
        for s in env.S:
            for t in env.Theta:
                principal_policy_noisy[s, t] = noisy_distribution(
                    principal_policy_optimal[s, t], self.alpha
                )

        agent = ObliviousRobustAgentA2C(env, principal_policy_optimal)
        agent.train()
        results["oblivious"] = agent.eval_episodes(principal_policy_noisy)

        agent = ObliviousRobustAgentPPO(env, principal_policy_optimal)
        agent.train()
        results["oblivious_PPO"] = agent.eval_episodes(principal_policy_noisy)

        # agent_bayesian = BayesianAgent(env, principal_policy_optimal)
        # results["bayesian"] = agent_bayesian.eval_episodes(principal_policy_noisy)

        # agent = RobustAgentPPO(env)
        # agent.train()
        # results["robust_PPO"] = agent.eval_noisy_episodes()

        # agent = RobustAgentPPOWithPrincipal(env, principal_policy_optimal)
        # agent.train()
        # results["robust_PPO_with_principal"] = agent.eval_noisy_episodes(
        #     principal_policy_noisy
        # )

        # agent = RobustAgent(env)
        # agent.train()
        # results["robust_A2C"] = agent.eval_noisy_episodes()

        # agent = RobustAgentWithPrincipal(env, principal_policy_optimal)
        # agent.train()
        # results["robust_A2C_with_principal"] = agent.eval_noisy_episodes(
        #     principal_policy_noisy
        # )

        # agent = RobustAgentAction(env, principal_policy_optimal)
        # agent.train()
        # results["robust_action"] = agent.eval_noisy_episodes()

        # agent = RobustAgentActionPPO(env, principal_policy_optimal)
        # agent.train()
        # results["robust_action_PPO"] = agent.eval_noisy_episodes()

        # agent = BayesianAgentAction(env, principal_policy_optimal)
        # results["bayesian"] = agent.eval_episodes()

        return results


if __name__ == "__main__":
    game = Game()
    print(game.play())
