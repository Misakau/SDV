import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal, Normal

from util import mlp, Swish
from util import DEFAULT_DEVICE

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2,static_fns=None,act_fnc='relu'):
        super().__init__()
        if act_fnc == 'swish':
            activation = Swish
        else:
            activation = nn.ReLU
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],activation)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.static_fns = static_fns

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)
        # if mean.ndim > 1:
        #     batch_size = len(obs)
        #     return MultivariateNormal(mean, scale_tril=scale_tril.repeat(batch_size, 1, 1))
        # else:
        #     return MultivariateNormal(mean, scale_tril=scale_tril)
    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            return dist.mean if deterministic else dist.sample()
        
    @torch.no_grad()
    def predict(self, obs, act, obs_mean, obs_std, max_r, min_r, deterministic=False):
        """
        obs and act are normalized
        predict next_obs and reward
        """
        if len(obs.shape) == 1:
            obs = obs[None, ]
            act = act[None, ]
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(DEFAULT_DEVICE)
        if not isinstance(act, torch.Tensor):
            act = torch.FloatTensor(act).to(DEFAULT_DEVICE)
        
        model_input = torch.cat([obs, act], dim=-1).to(DEFAULT_DEVICE)
        dist = self(model_input)
        pred_samples = dist.mean if deterministic else dist.sample()

        next_obs, rewards = pred_samples[:, :-1], pred_samples[:, -1]

        next_obs = next_obs.detach().cpu().numpy()
        rewards = rewards.detach().cpu().numpy()
        rewards = np.clip(rewards, min_r,max_r)
        obs = obs.detach().cpu().numpy()
        act = act.detach().cpu().numpy()
        real_obs = obs_std * obs + obs_mean
        real_next_obs = obs_std * next_obs + obs_mean
        terminals = self.static_fns.termination_fn(real_obs, act, real_next_obs)
        penalized_rewards = rewards
        penalized_rewards = penalized_rewards[:, None]
        terminals = terminals[:, None]
        return next_obs, penalized_rewards, terminals
    
    @torch.no_grad()
    def get_mean_var(self, obs, act):
        """
        obs and act are not normalized
        predict next_obs and reward
        """
        if len(obs.shape) == 1:
            obs = obs[None, ]
            act = act[None, ]
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(DEFAULT_DEVICE)
        if not isinstance(act, torch.Tensor):
            act = torch.FloatTensor(act).to(DEFAULT_DEVICE)
        
        model_input = torch.cat([obs, act], dim=-1).to(DEFAULT_DEVICE)
        dist = self(model_input)
        pred_means , pred_var = dist.mean, dist.variance

        next_obs, rewards = pred_means[:, :-1], pred_means[:, -1]
        next_obs_var, rewards_var = pred_var[:, :-1], pred_var[:, -1]
        next_obs = next_obs.detach().cpu().numpy()
        rewards = rewards.detach().cpu().numpy()
        next_obs_var = next_obs_var.detach().cpu().numpy()
        rewards_var = rewards_var.detach().cpu().numpy()
        rewards_var = rewards_var[:, None]
        penalized_rewards = rewards
        penalized_rewards = penalized_rewards[:, None]
        return next_obs, next_obs_var, penalized_rewards, rewards_var
    
class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)

    def forward(self, obs):
        return self.net(obs)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs)