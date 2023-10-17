import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
from util import torchify
class FakeEnv:
    def __init__(self, name, max_size=10**5,obs_dim=1,act_dim=1):
        self.name = name
        self.size = max_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs = np.ones((max_size,obs_dim),dtype=np.float32) * 0.78
        self.terminals = np.ones(max_size,dtype=np.float32)
        self.acts = np.zeros((max_size,act_dim),dtype=np.float32)
        self.next_obs = np.zeros((max_size,obs_dim),dtype=np.float32)
        self.rewards = np.zeros(max_size,dtype=np.float32)
        self.std = 0.01
        self.dataset = {}
        self.action_space = None
    def make(self, mode='continue'):
        # action: -1,1
        # obs: -2~2
        if mode == 'continue':
            eps = 0.001
            max_act = 0.3
            min_act = -0.1
            batch_size = int(self.size / ((max_act - min_act) / eps))
            cur_act = min_act
            ptr = 0
            while cur_act <= 0.3:
                obs_mean = cur_act + 0.09 / (cur_act + 0.3) - 0.35
                myobs = np.random.normal(obs_mean,self.std,size=batch_size).astype(np.float32)
                self.rewards[ptr:ptr + batch_size] = myobs.copy()
                myobs = myobs[:,None]
                myact = (np.ones((batch_size,self.act_dim)) * cur_act).astype(np.float32)
                assert(myobs.shape==(batch_size,self.obs_dim))
                assert(myact.shape==(batch_size,self.act_dim))
                self.next_obs[ptr:ptr + batch_size] = myobs.copy()
                self.acts[ptr:ptr + batch_size] = myact.copy()
                ptr += batch_size
                cur_act += eps
        else:
            acts = [-0.5, -0.1, 0.3, 0.5]
            means = [-0.3, 0.1, 0.7, 0.3]
            self.std = 0.1
            batch_size = int(self.size / len(acts))
            ptr = 0
            for i in range(len(acts)):
                obs_mean = means[i]
                myobs = np.random.normal(obs_mean,self.std,size=batch_size).astype(np.float32)
                self.rewards[ptr:ptr + batch_size] = myobs.copy()
                myobs = myobs[:,None]
                myact = (np.ones((batch_size,self.act_dim)) * acts[i]).astype(np.float32)
                assert(myobs.shape==(batch_size,self.obs_dim))
                assert(myact.shape==(batch_size,self.act_dim))
                self.next_obs[ptr:ptr + batch_size] = myobs.copy()
                self.acts[ptr:ptr + batch_size] = myact.copy()
                ptr += batch_size

    def get_dataset(self):
        self.dataset['observations'] = self.obs
        self.dataset['next_observations'] = self.next_obs
        self.dataset['actions'] = self.acts
        self.dataset['rewards'] = self.rewards
        self.dataset['terminals'] = self.terminals
        return deepcopy(self.dataset)
    
    def plot_data(self):
        plt.figure()
        plt.ylabel('T(s\'|s,a)')
        plt.xlabel('a')
        plt.scatter(self.acts[:,0],self.rewards,s=0.01)
        plt.savefig("./fake_env.png")
        plt.close()
    
    def seed(self,seed):
        pass

    def predict(self,goal_policy, dynamics_model, obs, rnobs):
        actions = np.arange(-0.5,0.5,0.001,dtype=np.float32)
        obs = obs[:len(actions)]
        rnobs = rnobs.detach().cpu().numpy()
        next_observations,next_observations_var, rewards, rewards_var = dynamics_model.get_mean_var(obs, actions.reshape(-1,1))
        #rewards reshape
        goal_obs = goal_policy.act(obs, deterministic=True).cpu().numpy()
        dists = (goal_obs - next_observations)*(goal_obs - next_observations)
        dists = np.sqrt(np.sum(dists,axis=1,keepdims=True))
        mystd = np.sqrt(np.sum(next_observations_var,axis=1,keepdims=True) + rewards_var)
        penalty = 0.04
        weights = dists
        s_rewards = rewards.copy()
        s_rewards = rewards - penalty * weights
        plt.figure()
        plt.ylabel('r(s\'|s,a)')
        plt.xlabel('a')
        plt.scatter(self.acts[:,0],self.rewards,s=0.01,c='r')
        plt.scatter(actions,rewards[:,0],s=0.01,c='b')
        plt.savefig("./fake_env_reward_pred.png")
        plt.close()

        plt.figure()
        plt.ylabel('T(s\'|s,a)')
        plt.xlabel('a')
        plt.scatter(self.acts[:,0],rnobs[:,0],s=0.01,c='r')
        plt.scatter(actions,next_observations[:,0],s=0.01,c='b')
        plt.scatter(actions,goal_obs[:,0],s=0.01,c='g')
        plt.savefig("./fake_env_obs_pred.png")
        plt.close()

        plt.figure()
        plt.ylabel('r(s\'|s,a)')
        plt.xlabel('a')
        plt.scatter(self.acts[:,0],self.rewards,s=0.01,c='r')
        plt.scatter(actions,s_rewards[:,0],s=0.01,c='b')
        plt.scatter(actions,rewards[:,0],s=0.01,c='g')
        plt.savefig("./fake_env_reward_reshape_pred.png")
        plt.close()
        # plt.scatter(actions,dists[:,0],s=0.01,c='orange')
        