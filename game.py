from environment import Environment
from robust_agent import RobustAgent

if __name__ == "__main__":
    env = Environment()
    agent = RobustAgent(env)
    for i in range(5):
        print(agent.state)
        agent.step()
