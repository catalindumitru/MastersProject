from environment import Environment
from robust_agent import RobustAgent


# def train_agent(agent):
#     env = agent.env
#     while True:
#         if config.max_steps and agent.total_steps >= config.max_steps:
#             agent.close()
#             break
#         agent.step()
#         agent.switch_task()
#     agent.save("data/%s-%s-%d" % (agent_name, config.tag, agent.total_steps))


if __name__ == "__main__":
    env = Environment()
    agent = RobustAgent(env)
    while agent.total_steps < agent.max_steps:
        agent.step()
