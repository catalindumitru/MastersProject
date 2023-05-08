from environment import Environment
from robust_agent import RobustAgent
from robust_agent_PPO import RobustAgentPPO
from bayesian_agent import BayesianAgent
from robust_agent_with_principal import RobustAgentWithPrincipal
from robust_agent_PPO_with_principal import RobustAgentPPOWithPrincipal
import torch

if __name__ == "__main__":
    env = Environment()

    # agent_bayesian = BayesianAgent(env)
    # agent_bayesian.train_get_policy()
    # print(agent_bayesian.eval_episodes())
    # print(agent_bayesian.eval_episodes_diff_policy("noisy"))
    # print(agent_bayesian.eval_episodes_diff_policy("random"))

    agent = RobustAgent(env)
    agent.train()
    print(agent.eval_noisy_episodes())

    # agent = RobustAgentWithPrincipal(env, agent_bayesian.optimal_policy)
    # agent.train()
    # print(agent.eval_noisy_episodes())

    # agent = RobustAgentPPO(env)
    # agent.train()
    # print(agent.eval_noisy_episodes())

    # agent = RobustAgentPPOWithPrincipal(env, agent_bayesian.optimal_policy)
    # agent.train()
    # print(agent.eval_noisy_episodes())
