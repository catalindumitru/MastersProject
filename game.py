from environment import Environment
from robust_agent import RobustAgent


def train_agent(agent):
    while agent.total_steps < agent.max_steps:
        agent.step()



if __name__ == "__main__":
    env = Environment()
    agent = RobustAgent(env)
    train_agent(agent)

    print(agent.eval_noisy_episodes())
