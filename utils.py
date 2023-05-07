import numpy as np
import torch.nn as nn
from torch import Tensor, from_numpy
from torch.distributions import Categorical


def random_distribution(n):
    d = np.random.rand(n)
    d = d / sum(d)
    return d


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[: len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def tensor(x):
    if isinstance(x, Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = from_numpy(x)
    return x


def to_np(t):
    return t.cpu().detach().numpy()


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def uniform_kernel(theta_count):
    return Categorical(tensor(random_distribution(theta_count))).sample().unsqueeze(0)


def kernel_without_principal(state, mu):
    return Categorical(tensor(mu[state])).sample().unsqueeze(0)
