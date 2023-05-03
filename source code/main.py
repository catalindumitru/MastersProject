import numpy as np
from mdpes import *
from time import strftime
import networkx as nx


# Generate a random distribution
def randDis(n):
    d = np.random.rand(n)
    d = d / sum(d)
    return d


# Generate a random instance
# lb/ub: lower/upper bound
def randInstance(num_S, num_Theta, num_A, num_termin=1, beta=0, lb_pr=0, ub_pr=1, lb_ag=0, ub_ag=1):
    m = MdpEs(num_S, num_Theta, num_A)
    m.genMethod = 'rand'
    m.initDis = randDis(num_S)
    for s in range(num_S):
        m.mu[s, :] = randDis(num_Theta)
        for a in range(num_A):
            m.P[s, a, :] = randDis(num_S)
    m.R_pr = np.random.uniform(low=lb_pr, high=ub_pr, size=(num_S, num_Theta, num_A))
    m.R_ag = np.random.uniform(low=lb_ag, high=ub_ag, size=(num_S, num_Theta, num_A))

    for t in range(num_S - num_termin + 1, num_S):
        for a in range(num_A):
            m.P[t, a, :] = 0
            m.P[t, a, t] = 1
        m.R_pr[t, :, :] = 0
        m.R_ag[t, :, :] = 0

    m.R_ag = (1 - np.abs(beta)) * m.R_ag + beta * m.R_pr

    return m


# Generate a random DAG
def randDAG(num_node, num_edge):
    T = nx.random_tree(num_node)
    M = nx.to_numpy_matrix(T)

    L = - np.ones(num_node)
    cur = 0
    q = [0]
    while q:
        i = q.pop(0)
        L[i] = cur
        cur += 1
        if cur == num_node:
            des = i
        for j in range(num_node):
            if M[i, j] == 1 and L[j] < 0:
                q.append(j)
                M[j, i] = 0

    _num_edge = num_node - 1

    while _num_edge < num_edge:
        i = np.random.randint(0, num_node)
        j = np.random.randint(0, num_node)

        if L[i] < L[j] and M[i, j] == 0:
            M[i, j] = 1
            _num_edge = _num_edge + 1
        elif L[i] > L[j] and M[j, i] == 0:
            M[j, i] = 1
            _num_edge = _num_edge + 1

    # swap the furthest node to the last row
    M[[des, num_node-1], :] = M[[num_node-1, des], :]
    M[:, [des, num_node - 1]] = M[:, [num_node - 1, des]]

    # connect remaining leaf nodes to the destination
    for i in range(num_node-1):
        if np.sum(M[i, :]) == 0:
            M[i, num_node-1] = 1

    return M


# Generate a road network navigation instance on a DAG
def randInstance_roadNav(num_node, num_road, num_Theta=3, beta=0, mode='NONE'):
    num_S = num_node
    num_A = num_node

    m = MdpEs(num_S, num_Theta, num_A)
    m.genMethod = 'roadNav'
    if mode != 'NONE':
        m.genMethod += mode

    # Initially always start at s0
    m.initDis = np.zeros(num_S)
    m.initDis[0] = 1

    # Generate a random connected graph
    M = randDAG(num_node, num_road)

    # Construct reward and transition functions
    max_cost = 1E10  # Cost for moving to a disconnected node
    for i in range(num_node):
        for j in range(num_node):
            m.P[i, j, :] = 0
            m.P[i, j, j] = 1

            if M[i, j] == 1:
                m.R_ag[i, :, j] = - np.random.rand(num_Theta)
                m.R_pr[i, :, j] = - np.random.rand(num_Theta)

                if mode == 'uniform_theta':
                    m.R_ag[i, :, j] = np.sort(m.R_ag[i, :, j])

                m.R_ag[i, :, j] = (1 - np.abs(beta)) * m.R_ag[i, :, j] + beta * m.R_pr[i, :, j]
            else:
                m.R_ag[i, :, j] = - max_cost
                m.R_pr[i, :, j] = - max_cost

        if i == num_node - 1:
            m.R_ag[i, :, i] = 0
            m.R_pr[i, :, i] = 0
    return m


# =============================
# --         MAIN            --
# =============================
if __name__ == "__main__":

    # 0. Set parameters
    num_S = 10          # Number of states
    num_Theta = 10      # Number of external parameters
    num_A = 10          # Number of actions
    num_termin = 5      # Number of terminal states

    # Parameters for road navigation instance
    # num_node = 20       # Number of nodes
    # num_road = 100      # Number of roads

    # Discount factors
    gamma_pr = 0.8      # principal
    gamma_ag = 0.8      # agent

    # 1. Generate a random instance
    m = randInstance(num_S, num_Theta, num_A, num_termin, beta=0)
    # Or a random road navigation instance with the following code
    # m = randInstance_roadNav(num_node, num_road, num_Theta, beta=0, mode='THETA_SORTED')

    # 2. Compute payoffs of different strategies
    noSigMyop   = m.noSigMyop(gamma_pr)             # no signal + myopic agent
    noSigFS     = m.noSigFS(gamma_pr, gamma_ag)     # no signal + FS agent
    optSigMyop  = m.optSigMyop(gamma_pr)            # optimal signal against myopic agent
    optSigAM    = m.optSigAM(gamma_pr, gamma_ag)    # optimal signal against advice-myopic agent
    fullControl = m.fullControl(gamma_pr)           # when principal has full control

    # 3. Print results
    print('noSigMyop:   ', noSigMyop)
    print('noSigFS:     ', noSigFS)
    print('optSigMyop:  ', optSigMyop)
    print('optSigAM:    ', optSigAM)
    print('fullControl: ', fullControl)



