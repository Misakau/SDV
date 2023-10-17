import numpy as np
import torch
import torch.nn as nn
from util import torchify

from copy import deepcopy


class SACPolicy(nn.Module):
    def __init__(
        self, 
        actor, 
        critic1, 
        critic2,
        actor_lr, 
        critic_lr, 
        action_space,
        dist, 
        tau=0.005, 
        gamma=0.99, 
        alpha=0.2,
        device="cpu"
    ):
        super().__init__()

        self.actor = actor
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()

        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
        self.critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

        self.action_space = action_space
        self.dist = dist

        self._tau = tau
        self._gamma = gamma

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha
        
        self.__eps = np.finfo(np.float32).eps.item()

        self._device = device
        
    def get_gamma(self):
        return self._gamma

    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
    
    def _sync_weight(self):
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def forward(self, obs, deterministic=False):
        dist = self.actor.get_dist(obs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action)

        if self.action_space:
            action_scale = torch.tensor((self.action_space.high - self.action_space.low) / 2, device=action.device)
        else:
            action_scale = torch.tensor(1, device=action.device)
        squashed_action = torch.tanh(action)
        log_prob = log_prob - torch.log(action_scale * (1 - squashed_action.pow(2)) + self.__eps).sum(-1, keepdim=True)

        return squashed_action, log_prob

    def sample_action(self, obs, deterministic=False):
        action, _ = self(obs, deterministic)
        return action.cpu().detach().numpy()

    def act_logprob(self, obs, squashed_action):
        dist = self.actor.get_dist(obs)
        action = torch.arctanh(torchify(squashed_action))
        log_prob = dist.log_prob(action)

        action_scale = torch.tensor((self.action_space.high - self.action_space.low) / 2, device=action.device)
        
        log_prob = log_prob - torch.log(action_scale * (1 - squashed_action.pow(2)) + self.__eps).sum(-1, keepdim=True)
        return log_prob


    def learn(self, data):
        obs, actions, next_obs, terminals, rewards = data["observations"], \
            data["actions"], data["next_observations"], data["terminals"], data["rewards"]
        
        rewards = torch.as_tensor(rewards).to(self._device)
        terminals = torch.as_tensor(terminals).to(self._device)
        
        # update critic
        q1, q2 = self.critic1(obs, actions).flatten(), self.critic2(obs, actions).flatten()
        with torch.no_grad():
            next_actions, next_log_probs = self(next_obs)
            next_q = torch.min(
                self.critic1_old(next_obs, next_actions), self.critic2_old(next_obs, next_actions)
            ) - self._alpha * next_log_probs
            target_q = rewards.flatten() + self._gamma * (1 - terminals.flatten()) * next_q.flatten()
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs = self(obs)
        q1a, q2a = self.critic1(obs, a).flatten(), self.critic2(obs, a).flatten()
        actor_loss = (self._alpha * log_probs.flatten() - torch.min(q1a, q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self._sync_weight()

        result =  {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        
        return result

    def V_np(self,obs):
        with torch.no_grad():
            next_actions, next_log_probs = self(obs)
            next_V = torch.min(
                self.critic1(obs, next_actions), self.critic2(obs, next_actions)
            ) - self._alpha * next_log_probs
        return next_V.cpu().detach().numpy()

    def Q_np(self,obs,act):
        with torch.no_grad():
            next_q = torch.min(
                self.critic1(obs, act), self.critic2(obs, act)
            )
        return next_q.cpu().detach().numpy()
    
    def behavior_clone(self, data):
        obs, actions = data["observations"], data["actions"]

        # update actor
        log_probs = self.act_logprob(obs,actions)
        actor_loss = (log_probs.flatten()).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        result = actor_loss.item()
        
        return result
