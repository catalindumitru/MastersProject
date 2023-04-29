import numpy as np


def random_distribution(n):
    d = np.random.rand(n)
    d = d / sum(d)
    return d
