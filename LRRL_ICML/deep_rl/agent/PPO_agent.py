#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import torch.nn

from ..network import *
from ..component import *
from .BaseAgent import *
import collections
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete


class PPOAgent(BaseAgent):
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
        if config.shared_repr:
            self.opt = config.optimizer_fn(self.network.parameters())
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda step: 1 - step / config.max_steps)
        else:
            self.actor_opt = config.actor_opt_fn(self.network.actor_params)
            self.critic_opt = config.critic_opt_fn(self.network.critic_params)
            if config.decaying_lr:
                # anneal lr
                l = lambda f: 1 - f / self.config.max_steps
                self.lr_scheduler_policy = torch.optim.lr_scheduler.LambdaLR(self.actor_opt, lr_lambda=l)
                self.lr_scheduler_value = torch.optim.lr_scheduler.LambdaLR(self.critic_opt, lr_lambda=l)
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.is_discrete = isinstance(self.task.action_space, Discrete)
        if self.is_discrete:
            self.loss_bound = 0.5
        else:
            self.loss_bound = 5

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.feed(prediction)
            storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                         'mask': tensor(1 - terminals).unsqueeze(-1),
                         'state': tensor(states)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
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

        entries = storage.extract(['log_pi_a','action', 'v', 'ret', 'advantage', 'entropy', 'pi', 'state'])
        EntryCLS = entries.__class__
        entries = EntryCLS(*list(map(lambda x: x.detach(), entries)))
        entries.advantage.copy_((entries.advantage - entries.advantage.mean()) / entries.advantage.std())

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(entries.state.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                entry = EntryCLS(*list(map(lambda x: x[batch_indices], entries)))

                prediction = self.network(entry.state, entry.action)
                ratio = (prediction['log_pi_a'] - entry.log_pi_a).exp()
                obj = ratio * entry.advantage
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * entry.advantage
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['entropy'].mean()
                if config.value_clip != -1:
                    value_loss = 0.5 * (torch.clamp(entry.ret - prediction['v'],-config.value_clip,config.value_clip)).pow(2).mean()
                else:
                    value_loss = 0.5 * (entry.ret - prediction['v']).pow(2).mean()
                approx_kl = (entry.log_pi_a - prediction['log_pi_a']).mean()
                if config.shared_repr:
                    if self.lexico and self.total_steps > self.config.max_steps/3:
                        weights = self.compute_weights()
                        robust_loss = self.robust_loss(entry.state.detach(),entry.pi.detach())
                        self.recent_losses[0].append(-policy_loss.mean())
                        self.recent_losses[1].append(-robust_loss)
                        self.update_lagrange()
                        self.opt.zero_grad()
                        # print("Policy loss: ", policy_loss)
                        # print("robust loss: ", robust_loss)
                        (policy_loss + robust_loss*weights[1]/weights[0] +
                         value_loss).backward()
                        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                        self.opt.step()
                    else:
                        self.opt.zero_grad()
                        (policy_loss + value_loss).backward()
                        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                        self.opt.step()
                else:
                    if approx_kl <= 1.5 * config.target_kl:
                        if self.lexico and self.total_steps > self.config.max_steps/3:
                            weights = self.compute_weights()
                            robust_loss = self.robust_loss(entry.state,entry.pi)
                            self.recent_losses[0].append(-policy_loss.mean())
                            self.recent_losses[1].append(robust_loss)
                            self.actor_opt.zero_grad()
                            (policy_loss*weights[0] + robust_loss*weights[1]).backward()
                            self.actor_opt.step()
                        else:
                            self.actor_opt.zero_grad()
                            policy_loss.backward()
                            self.actor_opt.step()
                    self.critic_opt.zero_grad()
                    value_loss.backward()
                    self.critic_opt.step()

        if config.decaying_lr:
            if config.shared_repr:
                self.lr_scheduler.step()
            else:
                self.lr_scheduler_policy.step()
                self.lr_scheduler_value.step()

    def eval_step(self, state):
        prediction = self.network(self.config.state_normalizer(state))
        action = to_np(prediction['action'])
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
        # Update Lagrange parameters
        for i in range(1):
            self.mu[i] += self.eta[i] * (self.j[i] - self.tau*self.j[i] - self.recent_losses[i][-1])
            self.mu[i] = max(self.mu[i], 0.0)
            #self.tau *= 0.999

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
        return torch.clip(loss,-self.loss_bound,self.loss_bound)


class SAPPOAgent(BaseAgent):
    def __init__(self, config, noise=None):
        BaseAgent.__init__(self, config, noise=noise)
        self.config = config
        self.kppo = config.kppo
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.sgld_network = config.network_fn()
        self.sgld_network.load_state_dict(self.network.state_dict())
        if config.shared_repr:
            self.opt = config.optimizer_fn(self.network.parameters())
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda step: 1 - step / config.max_steps)
        else:
            self.actor_opt = config.actor_opt_fn(self.network.actor_params)
            self.critic_opt = config.critic_opt_fn(self.network.critic_params)
            if config.decaying_lr:
                # anneal lr
                l = lambda f: 1 - f / self.config.max_steps
                self.lr_scheduler_policy = torch.optim.lr_scheduler.LambdaLR(self.actor_opt, lr_lambda=l)
                self.lr_scheduler_value = torch.optim.lr_scheduler.LambdaLR(self.critic_opt, lr_lambda=l)
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.feed(prediction)
            storage.feed({'reward': tensor(rewards).unsqueeze(-1),
                         'mask': tensor(1 - terminals).unsqueeze(-1),
                         'state': tensor(states)})
            states = next_states
            self.total_steps += config.num_workers
        self.states = states
        prediction = self.network(states)
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

        entries = storage.extract(['log_pi_a','action', 'v', 'ret', 'advantage', 'entropy', 'pi', 'state'])
        EntryCLS = entries.__class__
        entries = EntryCLS(*list(map(lambda x: x.detach(), entries)))
        entries.advantage.copy_((entries.advantage - entries.advantage.mean()) / entries.advantage.std())

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(entries.state.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                entry = EntryCLS(*list(map(lambda x: x[batch_indices], entries)))

                prediction = self.network(entry.state, entry.action)
                ratio = (prediction['log_pi_a'] - entry.log_pi_a).exp()
                obj = ratio * entry.advantage
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * entry.advantage
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['entropy'].mean()
                if self.config.lexico and self.total_steps > self.config.max_steps/3:
                    sa_loss = self.kppo*self.sa_loss(entry.state,entry.pi)
                else:
                    sa_loss = 0
                if config.value_clip != -1:
                    value_loss = 0.5 * (torch.clamp(entry.ret - prediction['v'],-config.value_clip,config.value_clip)).pow(2).mean()
                else:
                    value_loss = 0.5 * (entry.ret - prediction['v']).pow(2).mean()
                approx_kl = (entry.log_pi_a - prediction['log_pi_a']).mean()
                if config.shared_repr:
                    self.opt.zero_grad()
                    (policy_loss + sa_loss+ value_loss).backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                    self.opt.step()
                else:
                    if approx_kl <= 1.5 * config.target_kl:
                        self.actor_opt.zero_grad()
                        (policy_loss+sa_loss).backward()
                        self.actor_opt.step()
                    self.critic_opt.zero_grad()
                    value_loss.backward()
                    self.critic_opt.step()
                self.sgld_network.load_state_dict(self.network.state_dict())

        if config.decaying_lr:
            if config.shared_repr:
                self.lr_scheduler.step()
            else:
                self.lr_scheduler_policy.step()
                self.lr_scheduler_value.step()

    def eval_step(self, state):
        prediction = self.network(self.config.state_normalizer(state))
        action = to_np(prediction['action'])
        if isinstance(self.task.action_space, Box):
            action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        return action

    def sa_loss(self,states,actions):
        eps = self.noise.bound
        adv_states = get_state_kl_bound_sgld(self.sgld_network,states,actions,eps,10,0)
        self.config.state_normalizer.set_read_only()
        target = self.network.actor(self.config.state_normalizer(adv_states))
        self.config.state_normalizer.unset_read_only()
        loss = torch.nn.KLDivLoss()
        return loss(target,actions.detach())

