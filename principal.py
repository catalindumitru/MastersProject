import numpy as np
import gurobipy as gp
from gurobipy import GRB
from environment import Environment


class Principal:
    def __init__(self, env: Environment):
        self.env = env
        self.optimal_strategy = None

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

        self.optimal_strategy = pi_P

    def get_optimal_strategy(self) -> np.ndarray:
        if self.optimal_strategy is None:
            raise Exception("Principal not trained yet")
        return self.optimal_strategy
