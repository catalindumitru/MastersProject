import numpy as np
import torch.nn as nn
import torch
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


def noisy_distribution(dist, alpha):
    noise = random_distribution(len(dist))
    noisy_dist = alpha * dist + (1 - alpha) * noise
    return noisy_dist


def uniform_kernel(theta_count):
    return Categorical(tensor(random_distribution(theta_count))).sample()


def kernel_with_principal(state, theta, env, principal_policy_noisy):
    principal_action = Categorical(
        tensor(principal_policy_noisy[state, theta])
    ).sample()

    kernel = np.zeros((env.theta_count))
    denominator = 0
    for t in env.Theta:
        denominator += (
            principal_policy_noisy[state, t, principal_action] * env.mu[state, t]
        )
    for t in env.Theta:
        kernel[t] = (
            principal_policy_noisy[state, t, principal_action] * env.mu[state, t]
        ) / denominator

    return Categorical(tensor(kernel)).sample()


def kernel_without_principal(state, mu):
    return Categorical(tensor(mu[state])).sample()


def sample_signal(principal_policy, state, theta):
    return Categorical(tensor(principal_policy[state, theta])).sample()


noise_min = -3
noise_max = 3


def disturb_signal(signal, alpha):
    noise = np.random.uniform(low=noise_min, high=noise_max)
    return tensor(np.clip(signal + alpha * noise, 0, 24)).to(torch.int)
