import numpy as np
import gurobipy as gp
from gurobipy import GRB
from environment import Environment
from torch.distributions import Categorical
from utils import tensor, random_distribution


class BayesianAgent:
    def __init__(self, env: Environment) -> None:
        self.env = env

        self.max_steps = 1000
        self.episode_count = 100

        self.total_steps = 0
        self.state = self.env.sample_state()

    def reset(self):
        self.total_steps = 0
        self.state = self.env.sample_state()

    def train_get_policy(self):
        state_count = self.env.state_count
        action_count = self.env.action_count
        theta_count = self.env.theta_count

        solver = gp.Model()
        solver.Params.LogToConsole = 0

        # add variables
        V = solver.addVars(state_count, lb=-float("inf"))
        I = solver.addVars(state_count, action_count, action_count, lb=0.0)
        J = solver.addVars(state_count, theta_count, lb=-float("inf"))
        K = solver.addVars(state_count, action_count, theta_count, lb=0.0)

        # add constraints
        solver.addConstrs(
            self.env.mu[s, theta] * self.env.R_P[s, a, theta]
            + gp.quicksum(
                self.env.gamma_P * self.env.mu[s, theta] * self.env.P[s, a, ss] * V[ss]
                for ss in self.env.S
            )
            == gp.quicksum(
                I[s, a, b]
                * self.env.mu[s, theta]
                * (self.env.R_A[s, b, theta] - self.env.R_A[s, a, theta])
                for b in self.env.U
            )
            + J[s, theta]
            - K[s, a, theta]
            for s in self.env.S
            for theta in self.env.Theta
            for a in self.env.U
        )
        solver.addConstrs(
            V[s] >= gp.quicksum(J[s, theta] for theta in self.env.Theta)
            for s in self.env.S
        )

        obj = gp.quicksum(V[s] * self.env.init_distribution[s] for s in self.env.S)
        solver.setObjective(obj, GRB.MINIMIZE)

        solver.optimize()
        V = [V[s].X for s in self.env.S]

        pi_P = np.zeros((state_count, theta_count, action_count))

        for s in self.env.S:
            solverr = gp.Model()
            solverr.Params.LogToConsole = 0

            pi = solverr.addVars(theta_count, action_count, lb=0.0, ub=1.0)

            solverr.addConstr(
                gp.quicksum(
                    (
                        self.env.R_P[s, a, theta]
                        + self.env.gamma_P
                        * gp.quicksum(self.env.P[s, a, ss] * V[ss] for ss in self.env.S)
                    )
                    * self.env.mu[s, theta]
                    * pi[theta, a]
                    for a in self.env.U
                    for theta in self.env.Theta
                )
                == V[s]
            )

            solverr.addConstrs(
                gp.quicksum(
                    self.env.mu[s, theta] * pi[theta, a] * self.env.R_A[s, a, theta]
                    for theta in self.env.Theta
                )
                >= gp.quicksum(
                    self.env.mu[s, theta] * pi[theta, a] * self.env.R_A[s, b, theta]
                    for theta in self.env.Theta
                )
                for a in self.env.U
                for b in self.env.U
            )

            solverr.addConstrs(
                gp.quicksum(pi[theta, a] for a in self.env.U) == 1
                for theta in self.env.Theta
            )

            obj = 1
            solverr.setObjective(obj, GRB.MINIMIZE)

            solverr.optimize()

            for theta in self.env.Theta:
                for a in self.env.U:
                    pi_P[s, theta, a] = pi[theta, a].X if pi[theta, a].X >= 0 else 0

        self.pi_P = pi_P

    def train(self):
        state_count = self.env.state_count
        action_count = self.env.action_count
        theta_count = self.env.theta_count

        solver = gp.Model()
        solver.Params.LogToConsole = 0

        # add variables
        V = solver.addVars(state_count, lb=-float("inf"))
        I = solver.addVars(state_count, action_count, action_count, lb=0.0)
        J = solver.addVars(state_count, theta_count, lb=-float("inf"))
        K = solver.addVars(state_count, action_count, theta_count, lb=0.0)

        # add constraints
        solver.addConstrs(
            self.env.mu[s, theta] * self.env.R_P[s, a, theta]
            + gp.quicksum(
                self.env.gamma_P * self.env.mu[s, theta] * self.env.P[s, a, ss] * V[ss]
                for ss in self.env.S
            )
            == gp.quicksum(
                I[s, a, b]
                * self.env.mu[s, theta]
                * (self.env.R_A[s, b, theta] - self.env.R_A[s, a, theta])
                for b in self.env.U
            )
            + J[s, theta]
            - K[s, a, theta]
            for s in self.env.S
            for theta in self.env.Theta
            for a in self.env.U
        )
        solver.addConstrs(
            V[s] >= gp.quicksum(J[s, theta] for theta in self.env.Theta)
            for s in self.env.S
        )

        obj = gp.quicksum(V[s] * self.env.init_distribution[s] for s in self.env.S)
        solver.setObjective(obj, GRB.MINIMIZE)

        solver.optimize()

        # Extract the principal's optimal policy
        pi_P = np.zeros((state_count, theta_count, action_count))

        # for s in self.env.S:
        #     for theta in self.env.Theta:
        #         action_values = []
        #         for a in self.env.U:
        #             value = (
        #                 sum(
        #                     I[s, a, b].x
        #                     * self.env.mu[s, theta]
        #                     * (self.env.R_A[s, a, theta] - self.env.R_A[s, a, theta])
        #                     for b in self.env.U
        #                 )
        #                 + J[s, theta].x
        #                 - K[s, a, theta].x
        #             )
        #             action_values.append(value)

        #         # Normalize action probabilities for the principal's policy
        #         action_probabilities_P = np.exp(-np.array(action_values))
        #         action_probabilities_P /= action_probabilities_P.sum()

        #         pi_P[s, theta, :] = action_probabilities_P

        for s in range(state_count):
            for theta in range(theta_count):
                for a in range(action_count):
                    sum_I = sum(
                        I[s, a, b].X
                        * (self.env.R_A[s, b, theta] - self.env.R_A[s, a, theta])
                        * self.env.mu[s, theta]
                        for b in range(action_count)
                    )
                    sum_PV = sum(
                        self.env.gamma_P * self.env.P[s, a, ss] * V[ss].X
                        for ss in range(state_count)
                    )
                    pi_P[s, theta, a] = (
                        sum_I
                        + J[s, theta].X
                        - K[s, a, theta].X
                        - self.env.R_P[s, a, theta]
                        - sum_PV
                    ) / self.env.mu[s, theta]

        print(pi_P)
        self.pi_P = pi_P

    def eval_episode(self):
        self.reset()
        state = self.state
        total_reward_A = 0
        total_reward_P = 0
        while self.total_steps < self.max_steps:
            # print(self.total_steps)
            theta = self.env.sample_theta(state)
            signaled_action = Categorical(tensor(self.pi_P[state, theta, :])).sample()
            total_reward_A += self.env.R_A[state, signaled_action, theta]
            total_reward_P += self.env.R_P[state, signaled_action, theta]

            state = self.env.take_action(state, signaled_action)
            self.total_steps += 1

        return total_reward_A, total_reward_P

    def eval_episodes(self):
        episodic_rewards_A = []
        episodic_rewards_P = []
        for ep in range(self.episode_count):
            # print("Ep: ", ep)
            total_rewards_A, total_rewards_P = self.eval_episode()
            episodic_rewards_A.append(np.sum(total_rewards_A))
            episodic_rewards_P.append(np.sum(total_rewards_P))

        return np.mean(episodic_rewards_A), np.mean(episodic_rewards_P)

    def eval_episodes_diff_policy(self):
        for s in self.env.S:
            for theta in self.env.Theta:
                self.pi_P[s, theta, :] = random_distribution(self.env.action_count)
        episodic_rewards_A = []
        episodic_rewards_P = []
        for ep in range(self.episode_count):
            # print("Ep: ", ep)
            total_rewards_A, total_rewards_P = self.eval_episode()
            episodic_rewards_A.append(np.sum(total_rewards_A))
            episodic_rewards_P.append(np.sum(total_rewards_P))

        return np.mean(episodic_rewards_A), np.mean(episodic_rewards_P)
