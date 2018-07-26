"""
Parallel implementation of the Augmented Random Search method.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
"""

# FIXME(ev) make the shift part actually work
# FIXME(ev) test this code on a few examples
# FIXME(ev) import Linear Models in a way compliant with RLlib
# FIXME(ev) doesn't work for pendulum yet
# FIXME(ev) do rollout length in a standard way

import time
import os
import pickle
import numpy as np
# import gym
# import ray
from ray.rllib.es import utils
from ray.rllib.ars import optimizers
from ray.rllib import agent
from collections import namedtuple
from ray.rllib.ars.policies import *
from ray.rllib.es import tabular_logger as tlogger
import socket
import ray.tune as tune
from ray.tune import grid_search

Result = namedtuple("Result", [
    "noise_indices", "noisy_returns", "sign_noisy_returns", "noisy_lengths",
    "eval_returns", "eval_lengths"
])

DEFAULT_CONFIG = dict(
    num_workers=2,
    num_deltas=320,  # 320
    deltas_used=320,  # 320
    delta_std=0.02,
    sgd_stepsize=0.01,
    shift=0,
    observation_filter='NoFilter',
    policy='Linear',
    seed=123,
    eval_rollouts=50,
    env_config={}
)

DELTA_SIZE = 1e-6


@ray.remote
class Worker(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self, registry, config, env_creator,
                 env_seed,
                 # deltas=None,
                 rollout_length=1000,
                 delta_std=0.02):

        # initialize OpenAI environment for each worker
        self.env = env_creator(config["env_config"])
        self.env.seed(env_seed)

        from ray.rllib import models
        self.preprocessor = models.ModelCatalog.get_preprocessor(
            registry, self.env)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table. 
        # self.deltas = SharedNoiseTable(deltas, env_seed + 7)

        from ray.rllib import models
        self.preprocessor = models.ModelCatalog.get_preprocessor(
            registry, self.env)

        self.delta_std = delta_std
        self.rollout_length = rollout_length
        self.sess = utils.make_session(single_threaded=True)
        if config['policy'] == 'Linear':
            self.policy = LinearPolicy(
                registry, self.sess, self.env.action_space, self.preprocessor,
                config["observation_filter"])
        else:
            self.policy = MLPPolicy(
                registry, self.sess, self.env.action_space, self.preprocessor,
                config["observation_filter"])

    def rollout(self, shift=0., rollout_length=None):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """

        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.
        steps = 0

        ob = self.env.reset()
        for i in range(rollout_length):
            # print('observation before acting is', ob)
            action = self.policy.compute(ob)
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if done:
                break

        return total_reward, steps

    # def do_rollouts(self, w_policy, num_rollouts=1, shift=1, evaluate=False):
    def do_rollouts(self, w_policy, worker_idx, delta_size, num_rollouts=1, shift=1, evaluate=False, sample=False):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, steps, deltas_idx = [], [], []

        delta_idx = worker_idx * num_rollouts  # new line

        # for i in range(num_rollouts):
        for i in range(delta_idx, delta_idx+num_rollouts):  # new line
            if evaluate:
                self.policy.set_weights(w_policy)
                deltas_idx.append(-1)

                # for evaluation we do not shift the rewards (shift = 0)
                # and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                time_limit = self.env.spec.timestep_limit
                reward, r_steps = self.rollout(shift=0.,
                                               rollout_length=time_limit)
                rollout_rewards.append(reward)

            else:
                # idx, delta = self.deltas.get_delta(w_policy.size)
                # delta = (self.delta_std * delta).reshape(w_policy.shape)
                # deltas_idx.append(idx)
                delta = make_elementary_vector(i, w_policy.shape, delta_size)  # new line
                deltas_idx.append(i)  # new line

                # compute reward and number of timesteps used
                # for positive perturbation rollout
                self.policy.set_weights(w_policy + delta)
                # pos_reward, pos_steps = self.rollout(shift=shift)
                if not sample:  # new line
                    pos_reward, pos_steps = self.rollout(shift=shift)  # new line
                else:  # new line
                    # TODO(nskh) average reward over rollouts if stochastic  # new line
                    pos_reward, pos_steps = 0, 0  # new line
                    pass  # new line

                # compute reward and number of timesteps used f
                # or negative perturbation rollout
                self.policy.set_weights(w_policy - delta)
                # neg_reward, neg_steps = self.rollout(shift=shift)
                if not sample:  # new line
                    neg_reward, neg_steps = self.rollout(shift=shift)  # new line
                else:  # new line
                    # TODO(nskh) average reward over rollouts if stochastic  # new line
                    neg_reward, neg_steps = 0, 0  # new line
                    pass  # new line
                steps += [pos_steps, neg_steps]

                rollout_rewards.append([pos_reward, neg_reward])

                # # idx, delta = self.deltas.get_delta(w_policy.size)
                #
                # # delta = (self.delta_std * delta).reshape(w_policy.shape)
                #
                # deltas_idx.append(idx)
                #
                # # compute reward and number of timesteps used
                # # for positive perturbation rollout
                # self.policy.set_weights(w_policy + delta)
                # pos_reward, pos_steps = self.rollout(shift=shift)
                #
                # # compute reward and number of timesteps used f
                # # or negative pertubation rollout
                # self.policy.set_weights(w_policy - delta)
                # neg_reward, neg_steps = self.rollout(shift=shift)
                # steps += [pos_steps, neg_steps]
                #
                # rollout_rewards.append([pos_reward, neg_reward])

        return {'deltas_idx': deltas_idx,
                'rollout_rewards': rollout_rewards,
                "steps": steps}

    def get_weights(self):
        return self.policy.get_weights()


class ARSAgent(agent.Agent):
    """ 
    Object class implementing the ARS algorithm.
    """
    _agent_name = "ARS"
    _default_config = DEFAULT_CONFIG
    _allow_unknown_subkeys = ["env_config"]

    def _init(self):

        env = self.env_creator(self.config["env_config"])
        from ray.rllib import models
        preprocessor = models.ModelCatalog.get_preprocessor(
            self.registry, env)

        self.timesteps = 0
        # self.num_deltas = self.config["num_deltas"]
        # self.deltas_used = self.config["deltas_used"]
        self.step_size = self.config["sgd_stepsize"]
        # self.delta_std = self.config["delta_std"]
        seed = self.config["seed"]
        self.shift = self.config["shift"]
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')

        # Create the shared noise table.
        # print("Creating shared noise table.")
        # noise_id = create_shared_noise.remote()
        # self.deltas = SharedNoiseTable(ray.get(noise_id), seed=seed + 3)

        # Create the actors.
        print("Creating actors.")
        self.num_workers = self.config["num_workers"]
        self.workers = [
            Worker.remote(
                self.registry, self.config, self.env_creator,
                seed + 7 * i,
                # deltas=noise_id,
                rollout_length=env.spec.max_episode_steps,)
                # delta_std=self.delta_std)
            for i in range(self.config["num_workers"])]

        self.episodes_so_far = 0
        self.timesteps_so_far = 0

        self.sess = utils.make_session(single_threaded=False)
        # initialize policy
        if self.config['policy'] == 'MLP':
            self.policy = MLPPolicy(
                self.registry, self.sess, env.action_space, preprocessor,
                self.config["observation_filter"])
        else:
            self.policy = LinearPolicy(
                self.registry, self.sess, env.action_space, preprocessor,
                self.config["observation_filter"])
        self.w_policy = self.policy.get_weights()
        self.num_deltas = self.w_policy.size

        # initialize optimization algorithm
        self.optimizer = optimizers.SGD(self.w_policy, self.config["sgd_stepsize"])
        print("Initialization of ARS complete.")

    # FIXME(ev) should return the rewards and some other statistics
    # def aggregate_rollouts(self, num_rollouts=None, evaluate=False):
    def aggregate_rollouts(self, delta_size, num_rollouts=None, evaluate=False):  # new line (added delta_size)
        """ 
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts

        # put policy weights in the object store
        policy_id = ray.put(self.w_policy)

        t1 = time.time()
        num_rollouts = int(num_deltas / self.num_workers)

        # parallel generation of rollouts
        rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                     delta_idx,  # new line
                                                     delta_size,  # new line
                                                     num_rollouts=num_rollouts,
                                                     shift=self.shift,
                                                     evaluate=evaluate)
                           for delta_idx, worker in enumerate(self.workers)]
                            # for worker in self.workers]

        remainder_workers = self.workers[:(num_deltas % self.num_workers)]
        remainder_idx_start = int(num_deltas / self.num_workers) * self.num_workers  # new line
        remainder_indices = range(remainder_idx_start, remainder_idx_start + len(remainder_workers))  # new line
        # handle the remainder of num_delta/num_workers
        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                     delta_idx,  # new line
                                                     delta_size,  # new line
                                                     num_rollouts=1,
                                                     shift=self.shift,
                                                     evaluate=evaluate)
                           for delta_idx, worker in zip(remainder_indices, remainder_workers)]
                            # for worker in self.workers

        # gather results 
        results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        rollout_rewards, deltas_idx, steps = [], [], []

        for result in results_one:
            if not evaluate:
                self.timesteps += np.sum(result["steps"])
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']
            steps += [result['steps']]

        for result in results_two:
            if not evaluate:
                self.timesteps += np.sum(result["steps"])
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']
            steps += [result['steps']]

        info_dict = {'deltas_idx': deltas_idx,
                     'rollout_rewards': rollout_rewards,
                     'steps': steps}
        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype=np.float64)

        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards

        # # select top performing directions if deltas_used < num_deltas
        # max_rewards = np.max(rollout_rewards, axis=1)
        # if self.deltas_used > self.num_deltas:
        #     self.deltas_used = self.num_deltas
        #
        # percentage = (1 - (self.deltas_used / self.num_deltas))
        # idx = np.arange(max_rewards.size)[
        #     max_rewards >= np.percentile(max_rewards, 100 * percentage)]
        # deltas_idx = deltas_idx[idx]
        # rollout_rewards = rollout_rewards[idx, :]

        # normalize rewards by their standard deviation
        rollout_rewards /= np.std(rollout_rewards)

        t1 = time.time()
        # aggregate rollouts to form the gradient used to compute SGD step

        # reward_diff is vector of positive diff reward minus negative diff reward
        reward_diff = rollout_rewards[:, 0] - rollout_rewards[:, 1]
        deltas_tuple = np.array([make_elementary_vector(idx, self.w_policy.size, delta_size)  # new line
                                 for idx in deltas_idx])  # new line
        # deltas_tuple = (self.deltas.get(idx, self.w_policy.size)
        #                 for idx in deltas_idx)
        g_hat, count = utils.batched_weighted_sum(reward_diff, deltas_tuple,
                                                  batch_size=500)
        g_hat /= deltas_idx.size
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat, info_dict

    def train_step(self):
        """ 
        Perform one update step of the policy weights.
        """

        # grads = []
        # for exp in range(-1, -9, -1):  # new line
        #     g_hat, _ = self.aggregate_rollouts(delta_size=10**exp)  # new line
        #     grads.append(g_hat)
        # diffs = []
        # for i in range(len(grads)-1):
        #     diffs.append(np.linalg.norm(grads[i+1] - grads[i]))
        # print('\n\n\n***gradients***')
        # print('gradients:\n', grads)
        # print('norm of differences:\n', diffs)
        # print('***testing done***\n\n\n')
        g_hat, info_dict = self.aggregate_rollouts(delta_size=DELTA_SIZE)
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        compute_step = self.optimizer._compute_step(g_hat)
        self.w_policy -= compute_step.reshape(self.w_policy.shape)
        self.policy.set_weights(self.w_policy)
        return g_hat, info_dict

    def _train(self):

        # perform the training
        t1 = time.time()
        g_hat, info_dict = self.train_step()
        t2 = time.time()
        print('total time of one step', t2 - t1)

        self.episodes_so_far += len(info_dict['steps'])
        self.timesteps_so_far += np.sum(info_dict['steps'])

        # Evaluate the reward with the unperturbed params
        rewards = self.aggregate_rollouts(delta_size=DELTA_SIZE,
                                          num_rollouts=self.config['eval_rollouts'],
                                          evaluate=True)
        w = ray.get(self.workers[0].get_weights.remote())

        tlogger.record_tabular("AverageReward", np.mean(rewards))
        tlogger.record_tabular("StdRewards", np.std(rewards))
        tlogger.record_tabular("WeightNorm", float(np.square(w).sum()))
        tlogger.record_tabular("WeightStd", float(np.std(w)))
        tlogger.record_tabular("GradNorm", float(np.square(g_hat).sum()))
        tlogger.record_tabular("MaxRewardRollout", np.max(rewards))
        tlogger.record_tabular("MinRewardRollout", np.min(rewards))
        tlogger.dump_tabular()

        result = ray.tune.result.TrainingResult(
            episode_reward_mean=np.mean(rewards),
            episode_len_mean=np.mean(info_dict['steps']),
            timesteps_this_iter=np.sum(info_dict['steps']))

        return result

    def _stop(self):
        # workaround for https://github.com/ray-project/ray/issues/1516
        for w in self.workers:
            w.__ray_terminate__.remote(w._ray_actor_id.id())

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(
            checkpoint_dir, "checkpoint-{}".format(self.iteration))
        weights = self.policy.get_weights()
        objects = [
            weights,
            self.episodes_so_far,
            self.timesteps_so_far]
        pickle.dump(objects, open(checkpoint_path, "wb"))
        return checkpoint_path

    def _restore(self, checkpoint_path):
        objects = pickle.load(open(checkpoint_path, "rb"))
        self.policy.set_weights(objects[0])
        self.episodes_so_far = objects[1]
        self.timesteps_so_far = objects[2]

    def compute_action(self, observation):
        return self.policy.compute(observation, update=False)[0]


# new lines
def make_elementary_vector(idx, shape, step_size=1e-5):
    vec = np.zeros(shape)
    vec[idx] = 1.0*step_size
    return vec


if __name__ == '__main__':
    local_ip = socket.gethostbyname(socket.gethostname())
    ray.init(num_cpus=2, redirect_output=False)  # redis_address= local_ip + ':6379')

    # run_ars(params)
    config = DEFAULT_CONFIG
    config["step_size"] = grid_search([.02])
    tune.run_experiments({
        "my_experiment": {
            "run": "ARS",
            "stop": {
                "training_iteration": 10
            },
            "env": 'HalfCheetah-v2',
            "config": config,
            "trial_resources": {
                "cpu": 1,
                "gpu": 0,
                "extra_cpu": 1,
            }
        }
    })
