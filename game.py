from environment import Environment
from principal import Principal
from robust_agent import RobustAgent
from robust_agent_PPO import RobustAgentPPO
from bayesian_agent import BayesianAgent
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

        principal_policy_optimal = principal.get_optimal_policy()
        principal_policy_noisy = np.zeros(
            (env.state_count, env.theta_count, env.action_count)
        )
        for s in env.S:
            for t in env.Theta:
                principal_policy_noisy[s, t] = noisy_distribution(
                    principal_policy_optimal[s, t], self.alpha
                )

        agent_bayesian = BayesianAgent(env, principal_policy_noisy)
        results["bayesian"] = agent_bayesian.eval_episodes()

        agent = RobustAgent(env)
        agent.train()
        results["robust_A2C"] = agent.eval_noisy_episodes()

        agent = RobustAgentWithPrincipal(env, principal_policy_optimal)
        agent.train()
        results["robust_A2C_with_principal"] = agent.eval_noisy_episodes(
            principal_policy_noisy
        )

        agent = RobustAgentPPO(env)
        agent.train()
        results["robust_PPO"] = agent.eval_noisy_episodes()

        agent = RobustAgentPPOWithPrincipal(env, principal_policy_optimal)
        agent.train()
        results["robust_PPO_with_principal"] = agent.eval_noisy_episodes(
            principal_policy_noisy
        )

        return results


if __name__ == "__main__":
    game = Game()
    game.play()
