import gurobipy as gp
from gurobipy import GRB
import numpy as np
from copy import deepcopy


class Mdp(object):
    def __init__(self, num_S, num_A):
        self.num_S = num_S
        self.num_A = num_A
        self.S = range(self.num_S)
        self.A = range(self.num_A)

        self.initDis = np.ones(num_S) / num_S
        self.P = np.ones((num_S, num_A, num_S)) / num_S
        self.R = np.zeros((num_S, num_A))

        # Compute optimal value function of an MDP
    def optV(self, gamma):
        S, A = self.S, self.A

        solver = gp.Model()

        # add variables
        V = solver.addVars(self.num_S, lb=-float('inf'))
        Q = solver.addVars(self.num_S, self.num_A, lb=-float('inf'))

        # add constraints
        # 1. V(s) >= Q(s,a) for all a in A, s in S
        # 2. Q(s,a) >= R(s,a) + gamma P V for all a in A, s in S
        solver.addConstrs(
            V[s] >= Q[s, a]
            for a in A
            for s in S
        )
        solver.addConstrs(
            Q[s, a] ==
            self.R[s, a] + gp.quicksum(gamma * self.P[s, a, ss] * V[ss] for ss in S)
            for a in A
            for s in S
        )

        solver.setObjective(gp.quicksum(V[s] * self.initDis[s] for s in S), GRB.MINIMIZE)

        # solver.setParam(GRB.Param.DualReductions, 0)
        solver.optimize()

        return np.array([V[s].x for s in S]), np.array([[Q[s, a].x for a in A] for s in S])


# MDP with external state (external parameter)
class MdpEs(object):
    def __init__(self, num_S, num_Theta, num_A):
        self.num_S = num_S
        self.num_Theta = num_Theta
        self.num_A = num_A
        self.S = range(self.num_S)
        self.Theta = range(self.num_Theta)
        self.A = range(self.num_A)

        self.genMethod = ''
        self.initDis = np.ones(num_S) / num_S
        self.mu = np.ones((num_S, num_Theta)) / num_Theta
        self.P = np.ones((num_S, num_A, num_S)) / num_S
        self.R_pr = np.zeros((num_S, num_Theta, num_A))
        self.R_ag = np.zeros((num_S, num_Theta, num_A))

    # Generate MDP facing agent given no signal
    def agentMdpNoSig(self):
        m = Mdp(self.num_S, self.num_A)
        for s in self.S:
            for a in self.A:
                m.P[s, a, :] = self.P[s, a, :]
                m.R[s, a] = np.dot(self.R_ag[s, :, a], self.mu[s, :])
        return m

    # Generate equivalent MDP-ES when agent is advice myopic
    def eqMdpEsAM(self, gamma_ag):
        m = self.agentMdpNoSig()
        V, Q = m.optV(gamma_ag)
        _me = deepcopy(self)
        for s in self.S:
            for theta in self.Theta:
                for a in self.A:
                    _me.R_ag[s, theta, a] = self.R_ag[s, theta, a] + gamma_ag * np.dot(self.P[s, a, :], V)
        return _me

    # Principal's optimal payoff when agent is myopic
    def optSigMyop(self, gamma):
        S, Theta, A = self.S, self.Theta, self.A

        solver = gp.Model()

        # add variables
        V = solver.addVars(self.num_S, lb=-float('inf'))
        # pi = solver.addVars(num_S, num_Theta, num_A)
        I = solver.addVars(self.num_S, self.num_A, self.num_A, lb=0.0)
        J = solver.addVars(self.num_S, self.num_Theta, lb=-float('inf'))
        K = solver.addVars(self.num_S, self.num_A, self.num_Theta, lb=0.0)

        # add constraints
        # 1. dual constraints
        # 2. V(s) >= min dual obj
        solver.addConstrs(
            self.mu[s, theta] * self.R_pr[s, theta, a]
            + gp.quicksum(gamma * self.mu[s, theta] * self.P[s, a, ss] * V[ss] for ss in S)
            ==
            gp.quicksum(I[s, a, b] * self.mu[s, theta] * (self.R_ag[s, theta, b] - self.R_ag[s, theta, a]) for b in A)
            + J[s, theta] - K[s, a, theta]
            for s in S
            for theta in Theta
            for a in A
        )
        solver.addConstrs(
            V[s] >= gp.quicksum(J[s, theta] for theta in Theta)
            for s in S
        )

        obj = gp.quicksum(V[s] * self.initDis[s] for s in S)
        solver.setObjective(obj, GRB.MINIMIZE)

        #solver.setParam(GRB.Param.DualReductions, 0)
        solver.optimize()

        return obj.getValue()

    # Principal's optimal payoff when agent is advice-myopic
    def optSigAM(self, gamma_pr, gamma_ag):
        return self.eqMdpEsAM(gamma_ag).optSigMyop(gamma_pr)

    # Principal's payoff with full control over agent
    def fullControl(self, gamma_pr):
        S, Theta, A = self.S, self.Theta, self.A

        solver = gp.Model()

        m = Mdp(self.num_S * self.num_Theta, self.num_A)
        for s in S:
            for theta in Theta:
                s_theta = s * self.num_Theta + theta
                m.initDis[s_theta] = self.initDis[s] * self.mu[s, theta]
                for a in A:
                    m.R[s_theta, a] = self.R_pr[s, theta, a]
                    for ss in S:
                        for ttheta in Theta:
                            ss_ttheta = ss * self.num_Theta + ttheta
                            m.P[s_theta, a, ss_ttheta] = self.P[s, a, ss] * self.mu[ss, ttheta]
        V, Q = m.optV(gamma_pr)

        return np.dot(V, m.initDis)

    # Principal's payoff when no signal can be given
    def  noSigFS(self, gamma_pr, gamma_ag, eps=1E-5):
        S, Theta, A = self.S, self.Theta, self.A

        # Construct MDPs when no signaling
        m_ag = self.agentMdpNoSig()
        V_ag, Q_ag = m_ag.optV(gamma_ag)

        # MDP for principal
        m_pr = Mdp(self.num_S, self.num_A)
        for s in S:
            for a in A:
                m_pr.P[s, a, :] = self.P[s, a, :]
                if Q_ag[s, a] >= V_ag[s] - eps:
                    m_pr.R[s, a] = np.dot(self.R_pr[s, :, a], self.mu[s, :])
                else:
                    m_pr.R[s, a] = - 1E10 #float('inf')
        V_pr, Q_pr = m_pr.optV(gamma_pr)

        return np.dot(V_pr, self.initDis)

    # Principal's payoff when no signal can be given (against FS agent)
    def noSigMyop(self, gamma_pr, eps=1E-5):
        return self.noSigFS(gamma_pr, 0, eps)
