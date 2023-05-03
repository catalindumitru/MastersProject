#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import collections
from gym.spaces.box import Box


class A2CAgent(BaseAgent):
    def __init__(self, config, noise=None):
        BaseAgent.__init__(self, config, noise=noise)
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
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(config.state_normalizer(states))
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.feed(prediction)
            storage.feed({'state': tensor(states)})
            storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                         'mask': tensor(1 - terminals).unsqueeze(-1)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        storage.feed({'state': states})
        prediction = self.network(config.state_normalizer(states))
        storage.feed(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.reward[i] + config.discount * storage.mask[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + config.discount * storage.mask[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.mask[i] + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()
        entries = storage.extract(['log_pi_a', 'v', 'ret', 'advantage', 'entropy', 'pi', 'state'])
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
        entropy_loss = entries.entropy.mean()
        if self.lexico:
            weights = self.compute_weights()
            if self.total_steps>self.config.max_steps/3:
                robust_loss = self.robust_loss(entries.state,entries.pi)
                self.update_lagrange()
            else:
                robust_loss = tensor(0)
            self.recent_losses[0].append(-policy_loss)
            self.recent_losses[1].append(-robust_loss)
            self.optimizer.zero_grad()
            (policy_loss + robust_loss*weights[1]/weights[0] - config.entropy_weight * entropy_loss +
             config.value_loss_weight * value_loss).backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
            self.optimizer.step()
        else:
            self.optimizer.zero_grad()
            (policy_loss - config.entropy_weight * entropy_loss +
             config.value_loss_weight * value_loss).backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
            self.optimizer.step()

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        prediction = self.network(self.config.state_normalizer(state))
        action = to_np(prediction['action'])
        self.config.state_normalizer.unset_read_only()
        if isinstance(self.task.action_space, Box):
            action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        return action

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
            self.j[i] = torch.tensor(self.recent_losses[i]).mean()
        # Update Lagrange parameters, mu==lambda
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
        target = self.network.actor(self.config.state_normalizer(disturbed))
        self.config.state_normalizer.unset_read_only()
        loss = 0.5 * (actions.detach()-target).pow(2).mean()
        return torch.clip(loss,-0.2,0.2)
#
#
class QA2CAgent(BaseAgent):
    def __init__(self, config, noise=None):
        BaseAgent.__init__(self, config, noise=noise)
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
            self.tau = 0#.2
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length,keys=['pi','q','qa','eq'])
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(config.state_normalizer(states))
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.feed(prediction)
            storage.feed({'state': tensor(states)})
            storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                         'mask': tensor(1 - terminals).unsqueeze(-1)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        storage.feed({'state': states})
        prediction = self.network(config.state_normalizer(states))
        storage.feed(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['eq'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.reward[i] + config.discount * storage.mask[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.reward[i] + config.discount * storage.mask[i] * storage.eq[i + 1] - storage.eq[i]
                advantages = advantages * config.gae_tau * config.discount * storage.mask[i] + td_error
            storage.advantage[i] = advantages.detach()
            storage.ret[i] = returns.detach()
        entries = storage.extract(['log_pi_a', 'eq', 'ret', 'advantage', 'entropy', 'pi', 'state','qa','q'])
        policy_loss = -(entries.log_pi_a * entries.advantage).mean()
        value_loss = 0.5 * (entries.ret - entries.qa).pow(2).mean()
        entropy_loss = entries.entropy.mean()
        if self.lexico:
            weights = self.compute_weights()
            if self.total_steps > self.config.max_steps/3:
                # robust_loss = self.robust_Q_loss(entries.state,entries.q.detach(),entries.qa.detach(),entries.log_pi_a)
                robust_loss = self.robust_Q_loss(entries.state, entries.q.detach(), entries.ret,
                                                 entries.log_pi_a)
                # robust_loss = self.robust_loss(entries.state,entries.pi)
                self.update_lagrange()
            else:
                robust_loss = tensor(0)
            self.recent_losses[0].append(-policy_loss)
            self.recent_losses[1].append(robust_loss)
            self.optimizer.zero_grad()
            (policy_loss + robust_loss*weights[1]/weights[0] - config.entropy_weight * entropy_loss +
             config.value_loss_weight * value_loss).backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
            self.optimizer.step()
        else:
            self.optimizer.zero_grad()
            (policy_loss - config.entropy_weight * entropy_loss +
             config.value_loss_weight * value_loss).backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
            self.optimizer.step()

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        prediction = self.network(self.config.state_normalizer(state))
        action = to_np(prediction['action'])
        self.config.state_normalizer.unset_read_only()
        if isinstance(self.task.action_space, Box):
            action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        return action

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
        # Update Lagrange parameters, mu==lambda
        for i in range(1):
            self.mu[i] += self.eta[i] * (self.j[i] - self.tau*self.j[i] - self.recent_losses[i][-1])
            self.mu[i] = max(self.mu[i], 0.0)
            # self.tau *= 0.999

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

    def robust_Q_loss(self, states, q, qa, log_a):
        self.config.state_normalizer.set_read_only()
        disturbed = self.noise.nu(states)
        self.config.state_normalizer.unset_read_only()
        feats = self.network.feature(disturbed)
        probs = self.network.actor_phi(feats)
        q_pi_nu = torch.sum(probs["pi"] * q, dim=1).unsqueeze(-1)
        loss = (-qa+q_pi_nu).pow(2).mean()
        return torch.clip(loss, -0.2, 0.2)

    def robust_loss(self, states, actions):
        self.config.state_normalizer.set_read_only()
        disturbed = self.noise.nu(states)
        target = self.network.actor(self.config.state_normalizer(disturbed))
        self.config.state_normalizer.unset_read_only()
        loss = 0.5 * (actions.detach() - target).pow(2).mean()
        return torch.clip(loss, -0.2, 0.2)