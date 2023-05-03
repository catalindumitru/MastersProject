import numpy as np
import gym
import torch
from ..utils import *

# Noise Generator class. The noise generator in train mode can have different options based on what are the assumptions
# of the knowledge.


def make_noise(game, variance=0.5, bound=None, mode=0):
    env = gym.make(game)
    obs_space = env.observation_space
    return NoiseGenerator(obs_space,variance,bound,mode)


class NoiseGenerator:
    def __init__(self,obs_space, variance=0.5, bound=None, mode=0):
        self.obs_space = obs_space
        self.image = False
        try:
            if 'image' in obs_space.spaces.keys():
                self.image = True
                self.dtype = np.int8
                self.obs_space = obs_space['image']
                self.shape = obs_space.spaces['image'].shape
                self.low = obs_space.spaces['image'].low
                self.high = obs_space.spaces['image'].high
                self.var = variance
            else:
                self.dtype = np.float
                self.shape = obs_space.shape
                self.low = obs_space.low
                self.high = obs_space.high
                self.var = variance
        except:
            self.dtype = np.float
            self.shape = obs_space.shape
            self.low = obs_space.low
            self.high = obs_space.high
            self.var = variance
        self.mode = mode
        self.p_uniform = 0 # Probability of measuring true state in uniform noise
        if bound is None:
            if self.obs_space.is_bounded():
                self.bound = self.high.min()
                self.minbound = self.low.max()
            else:
                self.bound = 10 # random high bound for noise
                self.minbound = -10  # random high bound for noise
        else:
            self.bound = min(self.high.min(), bound)
            self.minbound = -self.bound
        self.k = 0.5*(self.bound-self.minbound)

    def set_noise(self,mode,bound=0,var=0):
        self.mode = mode
        if bound == 0:
            self.bound = np.clip(self.high*0.1,0,100)
            self.minbound = np.clip(self.low*0.1,-100,0)
        else:
            self.bound = bound
            self.minbound = -bound
        self.var = var

    def set_bound(self,bound):
        self.bound = bound

    def nu(self,x):
        if self.mode == 0:
            # # Uniform unbounded noise
            # if self.image:
            #     noise = np.zeros_like(x)
            #     for i, xi in enumerate(x):
            #         if np.random.uniform() > self.p_uniform:
            #             noise[i] = np.random.uniform(self.low, self.high, self.shape).astype(xi.dtype)
            #         else:
            #             noise[i] = xi
            # else:
            #     noise = np.zeros_like(x)
            #     for i,xi in enumerate(x):
            #         if np.random.uniform() > self.p_uniform:
            #             noise[i] = np.random.uniform(self.minbound, self.bound, self.shape)
            #         else: noise[i] = xi
            raise NotImplementedError

        elif self.mode == 1:
            # Uniform bounded noise
            noise = torch.rand_like(x)-0.5
            return tensor(x+2*self.bound*noise)
            # for i,xi in enumerate(x):
            #     noise[i] = np.clip(np.add(np.random.uniform(self.bound, self.minbound, self.shape),np.asarray(xi)),
            #                        self.low,self.high)if np.random.uniform() > self.p_uniform else xi
            # return torch.tensor(noise)
        elif self.mode == 2:
            x = to_np(x)
            noise = np.zeros_like(x)
            for i,xi in enumerate(x):
                noise[i] = np.clip(np.add(np.random.normal(0, self.var, size=self.shape)*np.mean(self.k),
                                          np.asarray(xi)),
                                   self.low,self.high).astype(xi.dtype)
            return tensor(noise)

    def adversarial_nu(self,x,policy,e=None):
        if e is None:
            e = self.bound
        actions = policy.actor(x)
        adv_states = get_state_kl_bound_sgld(policy, x, actions, e, 5, 0)
        return tensor(adv_states)

