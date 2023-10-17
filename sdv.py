import copy

from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

from util import DEFAULT_DEVICE, update_exponential_moving_average


EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class SDV(nn.Module):
    def __init__(self, qf, vf, goal_policy, goal_model, max_steps, max_model_steps,
                 tau, alpha, value_lr=1e-4, policy_lr=1e-4, discount=0.99, beta=0.005):
        super().__init__()
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.v_target = copy.deepcopy(vf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.goal_policy = goal_policy.to(DEFAULT_DEVICE)
        self.goal_model = goal_model.to(DEFAULT_DEVICE)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=value_lr)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=value_lr)
        self.goal_policy_optimizer = torch.optim.Adam(self.goal_policy.parameters(), lr=policy_lr)
        self.goal_model_optimizer = torch.optim.Adam(self.goal_model.parameters(), lr=policy_lr)
        self.goal_lr_schedule = CosineAnnealingLR(self.goal_policy_optimizer, max_model_steps)
        self.tau = tau
        self.alpha = alpha
        self.discount = discount
        self.beta = beta
        self.step = 0
        self.pretrain_step = 0

    def plot_env(self,observations,actions,next_observations,rewards):
        # goal_out = self.goal_model(torch.cat([observations,actions],dim=1))
        
        acts = torch.range(-1,1,0.01).view(-1,1).to(DEFAULT_DEVICE)
        print(acts.shape)
        iobs = observations[:acts.shape[0]]
        off_nobs = next_observations[:acts.shape[0]]
        print(iobs.shape)
        q_tar = self.q_target(iobs, acts)
        # output = goal_out.mean.detach().cpu().numpy()
        qout = q_tar.detach().cpu().numpy()
        # nobs = off_nobs.detach().cpu().numpy()
        # n_obs, _ = output[:,:-1], output[:,-1]
        plt.figure()
        plt.ylabel('Q_off(s,a)')
        plt.xlabel('a')
        myact = acts.detach().cpu().numpy()
        maxindx = qout.argmax()
        plt.plot(myact[:,0],qout,c='r')
        plt.plot([myact[maxindx],myact[maxindx]],[-0.03,-0.0125],c='black')
        plt.plot()
        plt.savefig("./fake_env_qoff_por.png")
        plt.close()

    def myiql_update(self, observations, actions, next_observations, rewards, terminals):
        # the network will NOT update
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v = self.vf(next_observations)
        
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        targets = rewards + (1. - terminals.float()) * self.discount * next_v
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.beta)

        # Update guidance (goal_policy)
        adv = targets - v
        galpha = self.alpha
        weight = torch.exp(galpha * adv)
        weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()
        goal_out = self.goal_policy(observations)
        g_loss = -goal_out.log_prob(next_observations)
        g_loss = torch.mean(weight * g_loss)
        self.goal_policy_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        self.goal_policy_optimizer.step()

        # Update model
        model_adv = targets - target_q
        inputs = torch.cat([observations,actions], 1)
        groundtruth = torch.cat([next_observations,rewards.view(-1,1)],dim=1)
        weight = torch.exp(self.alpha * model_adv)
        weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()
        goal_out = self.goal_model(inputs)
        g_m_loss = -goal_out.log_prob(groundtruth)

        g_m_loss = torch.mean(weight * g_m_loss)
        self.goal_model_optimizer.zero_grad(set_to_none=True)
        g_m_loss.backward()
        self.goal_model_optimizer.step()

        wandb.log({"g_loss": g_loss, "g_m_loss": g_m_loss}, step=self.step)

        # wandb
        if (self.step+1) % 10000 == 0:
            wandb.log({"v_loss": v_loss, "v_value": v.mean(), "q_loss": q_loss, "q_value": qs[0].mean()}, step=self.step)
        self.step += 1
    def mydouble_update(self, observations, actions, next_observations, rewards, terminals):
        # Update guidance (goal_policy)
        goal_out = self.goal_policy(observations)
        g_loss = -goal_out.log_prob(next_observations)
        g_loss = torch.mean(g_loss)
        self.goal_policy_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        self.goal_policy_optimizer.step()

        # Update model
        inputs = torch.cat([observations,actions], 1)
        groundtruth = torch.cat([next_observations,rewards.view(-1,1)],dim=1)
        goal_out = self.goal_model(inputs)
        g_m_loss = -goal_out.log_prob(groundtruth)

        g_m_loss = torch.mean(g_m_loss)
        self.goal_model_optimizer.zero_grad(set_to_none=True)
        g_m_loss.backward()
        self.goal_model_optimizer.step()

        wandb.log({"g_loss": g_loss, "g_m_loss": g_m_loss}, step=self.step)

        # wandb
        self.step += 1

    def save(self, filename):
        torch.save(self.goal_policy.state_dict(), filename + "-goal_network")
        # modify
        torch.save(self.goal_model.state_dict(), filename + "-model_network")
        print(f"***save models to {filename}***")

    def load(self, filename):
        self.goal_policy.load_state_dict(torch.load(filename + "-goal_network", map_location=DEFAULT_DEVICE))
        self.goal_model.load_state_dict(torch.load(filename + "-model_network", map_location=DEFAULT_DEVICE))
        print(f"***load the guidance and model from {filename}***")
