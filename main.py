"""

Implementation of SDV: Simple Double Validation Model-based Offline
Reinforcement Learning

This implementation builds upon 
1) a PyTorch reproduction of the implementation of MOPO:
   https://github.com/junming-yang/mopo.
2) Author's implementation of POR in "A Policy-Guided Imitation
   Approach for Offline Reinforcement Learning":
   https://github.com/ryanxhr/POR.

"""
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt

import gym
import os
import d4rl
import sys
import numpy as np
import torch
from tqdm import trange, tqdm

from sdv import SDV
from policy import GaussianPolicy
from value_functions import TwinQ, ValueFunction, TwinV
from util import return_range, set_seed, Log, sample_batch, torchify, evaluate_sdv
import wandb
import time

from util import DEFAULT_DEVICE
import importlib
from algo.buffer import ReplayBuffer
from algo.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from algo.mbpo import MBPO

def get_env_and_dataset(env_name, max_episode_steps, normalize):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        print(f'Dataset returns have range [{min_ret}, {max_ret}]')
        print(f'max score: {d4rl.get_normalized_score(args.env_name, max_ret) * 100.0}')
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps

    # dones = dataset["timeouts"]
    print("***********************************************************************")
    print(f"Normalize for the state: {normalize}")
    print("***********************************************************************")
    if normalize:
        mean = dataset['observations'].mean(0)
        std = dataset['observations'].std(0) + 1e-3
        dataset['observations'] = (dataset['observations'] - mean)/std
        dataset['next_observations'] = (dataset['next_observations'] - mean)/std
        if any(s in env_name for s in ('fake_env')):
            plt.figure()
            plt.ylabel('T(s\'|s,a)')
            plt.xlabel('a')
            plt.scatter(dataset['actions'][:,0],dataset['observations'][:,0],s=0.01)
            plt.scatter(dataset['actions'][:,0],dataset['rewards'],s=0.01)
            plt.savefig("./normalized_fake_env.png")
            plt.close()
    else:
        obs_dim = dataset['observations'].shape[1]
        mean, std = np.zeros(obs_dim), np.ones(obs_dim)

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset, mean, std

from fake_env import FakeEnv
def get_fake_env(env_name,max_episode_steps,normalize):
    fenv = FakeEnv(env_name)
    fenv.make(mode='d')
    fenv.plot_data()
    dataset = fenv.get_dataset()
    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        print(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['n_rewards'] = np.copy(dataset['rewards'])
        dataset['n_rewards'] /= (max_ret - min_ret)
        dataset['n_rewards'] *= max_episode_steps
    # dones = dataset["timeouts"]
    print("***********************************************************************")
    print(f"Normalize for the state: {normalize}")
    print("***********************************************************************")
    if normalize:

        if any(s in env_name for s in ('fake_env')):
            mean = dataset['next_observations'].mean(0)
            std = dataset['next_observations'].std(0)
            dataset['next_observations'] = (dataset['next_observations'] - mean)/std
        else:
            # assume that d_obs similar to d_next_obs
            mean = dataset['observations'].mean(0)
            std = dataset['observations'].std(0) + 1e-3
            dataset['observations'] = (dataset['observations'] - mean)/std
            dataset['next_observations'] = (dataset['next_observations'] - mean)/std
        print(mean)
        print(std)
    else:
        obs_dim = dataset['observations'].shape[1]
        mean, std = np.zeros(obs_dim), np.ones(obs_dim)

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return fenv, dataset, mean, std

def main(args):
    wandb.init(project="",
               entity="",
               name=f"{args.env_name}",
               config={
                   "env_name": args.env_name,
                   "normalize": args.normalize,
                   "tau": args.tau,
                   "alpha": args.alpha,
                   "seed": args.seed,
                   "type": args.type,
                   "value_lr": args.value_lr,
                   "policy_lr": args.policy_lr,
                   "pretrain": args.pretrain,
                   "sac_actor_lr": args.sac_actor_lr,
                   "sac_critic_lr": args.sac_critic_lr,
                   "sac_gamma": args.sac_gamma,
                   "sac_tau": args.sac_tau,
                   "sac_alpha_lr": args.sac_alpha_lr,
                   "rollout_length": args.rollout_length,
                   "real_ratio": args.real_ratio,
                   "reward_penalty_coef": args.reward_penalty_coef,
                   "beta": args.beta
               })
    torch.set_num_threads(1)

    if not args.fake_env:
        env, dataset, mean, std = get_env_and_dataset(args.env_name,
                                                    args.max_episode_steps,
                                                    args.normalize)
    else:
        env, dataset, mean, std = get_fake_env(args.env_name,
                                            args.max_episode_steps,
                                            args.normalize)
    r_max, r_min = dataset['rewards'].cpu().numpy().max(), dataset['rewards'].cpu().numpy().min()
    print(f"Single step rewards have range[{dataset['rewards'].min()},{dataset['rewards'].max()}]")
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    set_seed(args.seed, env=env)

    task = args.env_name.split('-')[0]
    import_path = f"static_fns.{task}"
    if not args.fake_env:
        static_fns = importlib.import_module(import_path).StaticFns
    else:
        static_fns = importlib.import_module("static_fns.halfcheetah").StaticFns
    goal_policy = GaussianPolicy(obs_dim, obs_dim, hidden_dim=args.hidden_dim, n_hidden=2)
    goal_model = GaussianPolicy(obs_dim + act_dim, obs_dim + 1, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,static_fns=static_fns,act_fnc='swish')

    # create buffer
    offline_buffer = ReplayBuffer(
        buffer_size=dataset["observations"].shape[0],
        obs_dim=obs_dim,
        obs_dtype=np.float32,
        act_dim=act_dim,
        act_dtype=np.float32
    )
    offline_buffer.load_dataset(dataset)
    model_buffer = ReplayBuffer(
        # buffer_size=args.rollout_batch_size * args.rollout_length * args.model_retain_epochs,
        buffer_size=10**6,
        obs_dim=obs_dim,
        obs_dtype=np.float32,
        act_dim=act_dim,
        act_dtype=np.float32
    )

    # create SAC policy model
    actor_backbone = MLP(input_dim=obs_dim, hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=obs_dim + act_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=obs_dim + act_dim, hidden_dims=[256, 256])
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=act_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    actor = ActorProb(actor_backbone, dist, DEFAULT_DEVICE)
    critic1 = Critic(critic1_backbone, DEFAULT_DEVICE)
    critic2 = Critic(critic2_backbone, DEFAULT_DEVICE)
    
    # auto-alpha
    sac_target_entropy = -act_dim
    sac_log_alpha = torch.zeros(1, requires_grad=True, device=DEFAULT_DEVICE)
    sac_alpha_optim = torch.optim.Adam([sac_log_alpha], lr=args.sac_alpha_lr)

    # create SAC policy
    sac_policy = SACPolicy(
        actor,
        critic1,
        critic2,
        actor_lr=args.sac_actor_lr,
        critic_lr=args.sac_critic_lr,
        action_space=env.action_space,
        dist=dist,
        tau=args.sac_tau,
        gamma=args.sac_gamma,
        alpha=(sac_target_entropy, sac_log_alpha, sac_alpha_optim),
        device=DEFAULT_DEVICE
    )
    

    sdv = SDV(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        vf=TwinV(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        goal_policy=goal_policy,
        goal_model=goal_model,
        max_steps=args.train_steps,
        max_model_steps=args.model_train_steps,
        tau=args.tau,
        alpha=args.alpha,
        discount=args.discount,
        value_lr=args.value_lr,
        policy_lr=args.policy_lr,
    )

    # train sdv
    if not args.pretrain:
        algo_name = f"{args.type}_alpha-{args.alpha}_tau-{args.tau}_alpha-{args.alpha}_normalize-{args.normalize}"
        os.makedirs(f"{args.log_dir}/{args.env_name}/{algo_name}", exist_ok=True)
        eval_log = open(f"{args.log_dir}/{args.env_name}/{algo_name}/seed-{args.seed}.txt", 'w')
        if not args.skip_model_train:
            # train guidance
            print(f"the models will be saved at {args.model_dir}/{args.env_name}/seed_{args.seed}_{algo_name}")
            for step in trange(args.model_train_steps):
                if args.type == 'sdv_f':
                    sdv.myiql_update(**sample_batch(dataset, args.batch_size))
                elif args.type == 'sdv_b':
                    # only double
                    sdv.mydouble_update(**sample_batch(dataset, args.batch_size))
                
            os.makedirs(f"{args.model_dir}/{args.env_name}", exist_ok=True)
            sdv.save(f"{args.model_dir}/{args.env_name}/seed_{args.seed}_{algo_name}")
        else:
            sdv.load(f"{args.model_dir}/{args.env_name}/seed_{args.seed}_{algo_name}")
            nobs , next_obs_var, _, rewards_var = sdv.goal_model.get_mean_var(dataset['observations'], dataset['actions'])
            gobs = sdv.goal_policy.act(dataset['observations'], deterministic=True).cpu().numpy()
            mydists = np.sqrt(np.mean((nobs - gobs) * (nobs - gobs), axis=1))
            mystds = np.sqrt(np.concatenate([next_obs_var,rewards_var],axis=1).mean(axis=1))
            print(np.min(mydists),np.max(mydists),np.mean(mydists))
            print(np.min(mystds),np.max(mystds),np.mean(mystds))

    if args.fake_env:
        sdv.plot_env(dataset['observations'],dataset['actions'],dataset['next_observations'],dataset['rewards'])
        env.predict(sdv.goal_policy,sdv.goal_model,dataset['observations'],dataset['next_observations'])
    mb_algo = MBPO(
        sac_policy,
        sdv.goal_model,
        sdv.goal_policy,
        obs_mean=mean,
        obs_std=std,
        r_max=r_max,
        r_min=r_min,
        beta = args.beta,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length,
        batch_size=256,
        real_ratio=args.real_ratio
    )

    def eval_sdv(step):
        eval_returns = np.array([evaluate_sdv(env, mb_algo, mean, std) \
                                 for _ in range(args.n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        print(f"return mean: {eval_returns.mean()},normalized return mean: {normalized_returns.mean()}")
        wandb.log({
            'return mean': eval_returns.mean(),
            'normalized return mean': normalized_returns.mean(),
        }, step=step)

        return normalized_returns.mean()
    
    if args.mle_test:
        mle_model = GaussianPolicy(obs_dim + act_dim, obs_dim + 1, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,static_fns=static_fns,act_fnc='swish').to(DEFAULT_DEVICE)
        mle_name = f"sdv_b_alpha-0.0_tau-{args.tau}_alpha-0.0_normalize-{args.normalize}"
        filename = f"{args.model_dir}/{args.env_name}/seed_{args.seed}_{mle_name}"
        mle_model.load_state_dict(torch.load(filename + "-model_network", map_location=DEFAULT_DEVICE))
        MLE_test(args,goal_model,mean,std,mle_model)

    # SAC
    if not args.pretrain:
        # train policy
        num_timesteps = 0
        train_epochs = 1000 if not args.fake_env else 100
        step_per_epoch = 1000 if not args.fake_env else 100
        for e in range(1, train_epochs + 1):
            if args.only_model:
                break
            # self.algo.model_buffer.clear() 
            mb_algo.policy.train()
            with tqdm(total=step_per_epoch, desc=f"Epoch #{e}/{train_epochs}") as t:
                while t.n < t.total:
                    if num_timesteps % args.rollout_freq == 0:
                        mb_algo.rollout_transitions(e)
                    # update policy by sac
                    loss = mb_algo.learn_policy()
                    t.set_postfix(**loss)
                    # log
                    """
                    if num_timesteps % args.log_freq == 0:
                        for k, v in loss.items():
                            self.logger.record(k, v, num_timesteps, printed=False)
                    """
                    num_timesteps += 1
                    t.update(1)
            # evaluate current policy
            if not args.fake_env:
                average_returns = eval_sdv(num_timesteps)
                eval_log.write(f'{num_timesteps + 1}\t{average_returns}\n')
                eval_log.flush()
                # save policy
                torch.save(mb_algo.policy.state_dict(), os.path.join(args.model_dir,args.env_name, "policy.pth"))
        if args.fake_env:
            mb_algo.plot_qs()    
    if not args.pretrain:
        eval_log.close()

def MLE_test(args, goal_model, mean, std, mle_model):
    from sklearn import manifold
    tsne = manifold.TSNE(random_state=1,n_iter=500)
    print("=====MLE TEST=====")
    my_env = gym.make(args.env_name)
    for le in [50]:
        print(f"========================LE: {le}========================")
        print("get object state...")
        obj_obs = my_env.reset()
        st = 0
        r_obj_obs = np.concatenate([my_env.sim.data.qpos,my_env.sim.data.qvel])
        last_obj = r_obj_obs
        ts = 0
        nqs={'hopper-random-v2':6,'halfcheetah-random-v2':9,'walker2d-random-v2':9}
        print(f"nq = {my_env.model.nq}, nv = {my_env.model.nv}")
        while st < le:
            obs, r, d, _ = my_env.step(my_env.action_space.sample())
            obj_obs = obs
            r_obj_obs = np.concatenate([my_env.sim.data.qpos,my_env.sim.data.qvel])
            if d:
                my_env.reset()
                myact = []
                ts+=1
                if ts % 100 == 0:
                    print(f"try {ts} times, st = {st}...")
                st = 0
            else: st+=1
            
        print("get real dist...")
        real_next_obs = []
        real_r = []
        # real_next_obs1 = []
        myact = []
        my_env.reset()
        with tqdm(total=10000, desc="Time") as t:
            while t.n < t.total:
                my_env.set_state(r_obj_obs[:nqs[args.env_name]],r_obj_obs[nqs[args.env_name]:])
                myact.append(my_env.action_space.sample())
                obs, r, d, _ = my_env.step(myact[-1])
                real_next_obs.append((obs-mean)/std)
                real_r.append(r)
                # real_next_obs1.append(obs[8])
                t.update(1)
        real_next_obs = np.array(real_next_obs)
        real_r = np.array(real_r)
        max_r, min_r = real_r.max(), real_r.min()
        real_r = (real_r - min_r) / (max_r - min_r)
        # print(len(real_next_obs))
        # print(real_next_obs[:4])
        print("get model dist...")
        model_next_obs = []
        # model_next_obs1 = []
        t_obj_obs = torchify(obj_obs).view(1,-1).repeat(10000,1)
        print(t_obj_obs.shape)
        t_act = torchify(np.array(myact))
        dist = goal_model(torch.concatenate([t_obj_obs,t_act],dim=1))
        mle_dist = mle_model(torch.concatenate([t_obj_obs,t_act],dim=1))
        ret = dist.sample().detach().cpu().numpy()
        mle_ret = mle_dist.sample().detach().cpu().numpy()
        obs, r = ret[:,:-1], ret[:,-1]
        # r /= max_r
        mle_obs, mle_r = mle_ret[:,:-1], mle_ret[:,-1]
        # mle_r /= max_r
        model_next_obs = (obs-mean)/std
        mle_model_next_obs = (mle_obs-mean)/std
        X_real = tsne.fit_transform(real_next_obs)
        X_model = tsne.fit_transform(model_next_obs)
        X_mle_model = tsne.fit_transform(mle_model_next_obs)
        plt.figure()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(X_real[:,0],X_real[:,1],s=1,alpha=real_r,label='real')
        plt.scatter(X_model[:,0],X_model[:,1],s=1,alpha=0.3,label='model')
        plt.legend()
        plt.savefig(f"{args.env_name}_sne_{le}_alpha{args.alpha}_sdv.png")
        plt.close()

        plt.figure()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(X_real[:,0],X_real[:,1],s=1,alpha=real_r,label='real')
        plt.scatter(X_mle_model[:,0],X_mle_model[:,1],s=1,alpha=0.3,label='MLE_model')
        plt.legend()
        plt.savefig(f"{args.env_name}_sne_{le}_alpha{args.alpha}_mle.png")
        plt.close()
            # model_next_obs1.append(obs[8])
        

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default="hopper-medium-replay-v2")
    parser.add_argument('--log_dir', type=str, default="./results/")
    parser.add_argument('--model_dir', type=str, default="./models/")
    parser.add_argument('--seed', type=int, default=1) #try 13
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=3)
    parser.add_argument('--pretrain_steps', type=int, default=5*10**5)
    parser.add_argument('--model_train_steps', type=int, default=5*10**5)
    parser.add_argument('--train_steps', type=int, default=10**6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--value_lr', type=float, default=1e-4)
    parser.add_argument('--policy_lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--eval_period', type=int, default=5000)
    parser.add_argument('--n_eval_episodes', type=int, default=10)
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--type", type=str, choices=['sdv_f', 'sdv_b'], default='sdv_f')
    parser.add_argument("--pretrain", action='store_true')
    parser.add_argument("--fake_env", action='store_true')
    parser.add_argument("--skip_model_train", action='store_true')
    parser.add_argument('--only_model', action='store_true')
    parser.add_argument('--mle_test', action='store_true')

    # SAC
    parser.add_argument("--sac_actor_lr", type=float, default=3e-4)
    parser.add_argument("--sac_critic_lr", type=float, default=3e-4)
    parser.add_argument("--sac_gamma", type=float, default=0.99)
    parser.add_argument("--sac_tau", type=float, default=0.005)
    parser.add_argument("--sac_alpha_lr", type=float, default=3e-4)

    # MBPO
    parser.add_argument("--rollout_freq", type=int, default=1000)
    parser.add_argument("--rollout_length", type=int, default=5)
    parser.add_argument("--real_ratio", type=float, default=0.5)
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument("--reward_penalty_coef", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0)
    now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args = parser.parse_args()
    
    main(args)