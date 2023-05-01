import numpy as np
import torch.nn as nn
from torch import Tensor, from_numpy


def random_distribution(n):
    d = np.random.rand(n)
    d = d / sum(d)
    return d


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


def uniform_kernel():
    pass
