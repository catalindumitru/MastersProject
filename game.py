from environment import Environment
from robust_agent import RobustAgent
from robust_agent_PPO import RobustAgentPPO
from bayesian_agent import BayesianAgent
import torch

if __name__ == "__main__":
    env = Environment()

    agent = RobustAgent(env)
    agent.train()
    print(agent.eval_noisy_episodes())

    # agent = RobustAgentPPO(env)
    # agent.train()
    # print(agent.eval_noisy_episodes())

    # agent = BayesianAgent(env)
    # agent.train_get_policy()
    # print(agent.eval_episodes())
    # print(agent.eval_episodes_diff_policy())
