from matplotlib import pyplot as plt
import numpy as np
from util import torchify

class MBPO:
    def __init__(
            self,
            policy,
            dynamics_model,
            goal_policy,
            obs_mean,
            obs_std,
            r_max,
            r_min,
            beta,
            offline_buffer,
            model_buffer,
            reward_penalty_coef,
            rollout_length,
            batch_size,
            real_ratio,
            rollout_batch_size=50000
    ):
        self.policy = policy
        self.goal_policy = goal_policy
        self.dynamics_model = dynamics_model
        self.offline_buffer = offline_buffer
        self.model_buffer = model_buffer
        self._reward_penalty_coef = reward_penalty_coef
        self._rollout_length = rollout_length
        self._rollout_batch_size = rollout_batch_size
        self._decay_rollout_batch_size = rollout_batch_size
        self._batch_size = batch_size
        self._real_ratio = real_ratio
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.r_max = r_max
        self.r_min = r_min
        self.beta = beta

    def _sample_initial_transitions(self):
        return self.offline_buffer.sample(self._decay_rollout_batch_size)

    def rollout_transitions(self, timestep):
        self._decay_rollout_batch_size =  self._rollout_batch_size
        init_transitions = self._sample_initial_transitions()
        # rollout
        observations = init_transitions["observations"]
        for _ in range(self._rollout_length):
            actions = self.policy.sample_action(observations)
            next_observations, rewards, terminals = self.dynamics_model.predict(observations, actions, self.obs_mean, self.obs_std, self.r_max, self.r_min)
            #rewards reshape
            goal_obs = self.goal_policy.act(torchify(observations), deterministic=True).cpu().numpy()
            dists = (goal_obs - next_observations)*(goal_obs - next_observations)
            dists = np.sqrt(np.sum(dists,axis=1,keepdims=True))

            _, obs_vars, _, rew_var = self.dynamics_model.get_mean_var(observations, actions)
            mystd = np.sqrt(np.sum(np.concatenate([obs_vars, rew_var],axis=1),axis=1,keepdims=True))
            # only dist
            penalty = dists
            rewards -= self._reward_penalty_coef * penalty

            self.model_buffer.add_batch(observations, next_observations, actions, rewards, terminals)
            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break
            observations = next_observations[nonterm_mask]

    def learn_policy(self):
        real_sample_size = int(self._batch_size * self._real_ratio)
        fake_sample_size = self._batch_size - real_sample_size
        real_batch = self.offline_buffer.sample(batch_size=real_sample_size)
        fake_batch = self.model_buffer.sample(batch_size=fake_sample_size)
        data = {
            "observations": np.concatenate([real_batch["observations"], fake_batch["observations"]], axis=0),
            "actions": np.concatenate([real_batch["actions"], fake_batch["actions"]], axis=0),
            "next_observations": np.concatenate([real_batch["next_observations"], fake_batch["next_observations"]],
                                                axis=0),
            "terminals": np.concatenate([real_batch["terminals"], fake_batch["terminals"]], axis=0),
            "rewards": np.concatenate([real_batch["rewards"], fake_batch["rewards"]], axis=0)
        }
        loss = self.policy.learn(data)
        return loss

    def clone_policy(self):
        real_batch = self.offline_buffer.sample(batch_size=self._batch_size)
        data = {
            "observations": real_batch["observations"],
            "actions": real_batch["actions"],
            "next_observations": real_batch["next_observations"],
            "terminals": real_batch["terminals"],
            "rewards": real_batch["rewards"],
        }
        loss = self.policy.behavior_clone(data)
        return loss
    
    def plot_qs(self):
        real_batch = self.offline_buffer.sample(batch_size=self._batch_size)
        act = np.arange(-1,1,0.01).reshape(-1,1)
        obs = real_batch["observations"][:act.shape[0]]
        qout = self.policy.Q_np(obs,act)
        plt.figure()
        plt.ylabel('Q_sac(s,a)')
        plt.xlabel('a')
        maxindx = qout.argmax()
        plt.plot(act[:,0],qout[:,0],c='r')
        plt.plot([act[maxindx],act[maxindx]],[-4,2],c='black')
        plt.plot()
        plt.savefig("./fake_env_qsac_por.png")
        plt.close()