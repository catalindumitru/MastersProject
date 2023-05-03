#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import torch

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision
import collections
from gym.spaces.discrete import Discrete


class DDPGAgent(BaseAgent):
    def __init__(self, config, noise=None):
        BaseAgent.__init__(self, config, noise=noise)
        self.config = config
        self.lexico = config.lexico
        if self.lexico:
            # Lexicographic Robustness
            self.i = None
            self.mu = [0.0 for _ in range(1)]
            self.j = [0.0 for _ in range(1)]
            self.recent_losses = [collections.deque(maxlen=25) for i in range(2)]
            self.beta = list(reversed(range(1, 3)))
            self.eta = [1e-3 * eta for eta in list(reversed(range(1, 3)))]
            self.second_order = False
            self.tau = 0.01
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.is_discrete = isinstance(self.task.action_space, Discrete)
        if self.is_discrete:
            self.loss_bound = 0.5
        else:
            self.loss_bound = 5
        self.state = None

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        if self.is_discrete:
            # action = torch.distributions.Categorical(probs=action).sample()
            action = epsilon_greedy(self.exploration, action['action'].detach())
            return action
        return to_np(action)

    def step(self):
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            if self.is_discrete:
                pi = F.normalize(torch.rand((1,self.task.action_dim)),p=1,dim=1)
                action = to_np(torch.argmax(pi).unsqueeze(0))
            else:
                action = [self.task.action_space.sample()]
                pi = None
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
            action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
            pi = None
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)

        self.replay.feed(dict(
            state=self.state,
            action=action,
            reward=reward,
            # pi=None,
            next_state=next_state,
            mask=1-np.asarray(done, dtype=np.int32),
        ))

        if done[0]:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if self.replay.size() >= config.warm_up:
            transitions = self.replay.sample()
            states = tensor(transitions.state)
            actions = tensor(transitions.action)
            rewards = tensor(transitions.reward).unsqueeze(-1)
            next_states = tensor(transitions.next_state)
            mask = tensor(transitions.mask).unsqueeze(-1)

            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = config.discount * mask * q_next
            q_next.add_(rewards)
            q_next = q_next.detach()
            phi = self.network.feature(states)
            q = self.network.critic(phi, actions)

            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            phi = self.network.feature(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()
            if self.lexico and self.total_steps > self.config.max_steps/3:
                weights = self.compute_weights()
                robust_loss = self.robust_loss(states,action)
                self.recent_losses[0].append(-policy_loss)
                self.recent_losses[1].append(robust_loss)
                self.network.zero_grad()
                (policy_loss + robust_loss * weights[1]/weights[0]).backward()
                # nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.network.actor_opt.step()
            else:
                self.network.zero_grad()
                policy_loss.backward()
                self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)

    def converged(self, tolerance=0.1, bound=0.01, minimum_updates=5):
        # If not enough updates have been performed, assume not converged
        if len(self.recent_losses[self.i]) < minimum_updates:
            return False
        else:
            l_mean = torch.tensor(self.recent_losses[self.i]).mean().float()
            # If the mean loss is lower than some absolute bound, assume converged
            if l_mean < bound:
                return True
            # Else check if the max of the recent losses are sufficiently close to the mean, if so then assume converged
            else:
                l_max = max(self.recent_losses[self.i]).float()
                if l_max > (1.0 + tolerance) * l_mean:
                    return False

        return True

    def update_lagrange(self):
        # Save relevant loss information for updating Lagrange parameters
        for i in range(1):
            self.j[i] = -torch.tensor(self.recent_losses[i]).mean()
        # Update Lagrange parameters
        for i in range(1):
            self.mu[i] += self.eta[i] * (self.j[i] - self.tau*self.j[i] - self.recent_losses[i][-1])
            self.mu[i] = max(self.mu[i], 0.0)

    def compute_weights(self):
        reward_range = 2
        # Compute weights for different components of the first order objective terms
        first_order = []
        for i in range(reward_range - 1):
            w = self.beta[i] + self.mu[i] * sum([self.beta[j] for j in range(i + 1, reward_range)])
            first_order.append(w)
        first_order.append(self.beta[reward_range - 1])
        first_order_weights = torch.tensor(first_order)
        return first_order_weights

    def robust_loss(self,states,actions):
        self.config.state_normalizer.set_read_only()
        disturbed = self.noise.nu(states)
        target = self.network.actor(self.network.feature(self.config.state_normalizer(disturbed)))
        self.config.state_normalizer.unset_read_only()
        loss = 0.5 * (actions.detach()-target).pow(2).mean()
        return torch.clip(loss,-self.loss_bound,self.loss_bound)
