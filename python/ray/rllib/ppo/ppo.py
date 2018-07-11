from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import ray
from ray.rllib.es import utils
from ray.tune.result import TrainingResult
from ray.rllib.agent import Agent
from ray.rllib.utils import FilterManager
from ray.rllib.ppo.ppo_evaluator import PPOEvaluator
from ray.rllib.ppo.rollout import collect_samples


DEFAULT_CONFIG = {
    # Discount factor of the MDP
    "gamma": 0.995,
    # Number of steps after which the rollout gets cut
    "horizon": 2000,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # GAE(lambda) parameter
    "lambda": 1.0,
    # Initial coefficient for KL divergence
    "kl_coeff": 0.2,
    # Number of SGD iterations in each outer loop
    "num_sgd_iter": 30,
    # Stepsize of SGD
    "sgd_stepsize": 5e-5,
    # TODO(pcm): Expose the choice between gpus and cpus as a command line argument.
    "devices": ["/cpu:%d" % i for i in range(4)],
    "tf_session_args": {
        "device_count": {"CPU": 4},
        "log_device_placement": False,
        "allow_soft_placement": True,
        "intra_op_parallelism_threads": 1,
        "inter_op_parallelism_threads": 2,
    },
    # Batch size for policy evaluations for rollouts
    "rollout_batchsize": 1,
    # Total SGD batch size across all devices for SGD
    "sgd_batchsize": 128,
    # Coefficient of the value function loss
    "vf_loss_coeff": 1.0,
    # Coefficient of the entropy regularizer
    "entropy_coeff": 0.0,
    # PPO clip parameter
    "clip_param": 0.3,
    # Target value for KL divergence
    "kl_target": 0.01,
    # Config params to pass to the model
    "model": {"free_log_std": False},
    # Which observation filter to apply to the observation
    "observation_filter": "MeanStdFilter",
    # If >1, adds frameskip
    "extra_frameskip": 1,
    # Number of timesteps collected in each outer loop
    "timesteps_per_batch": 4000,
    # Each tasks performs rollouts until at least this
    # number of steps is obtained
    "min_steps_per_task": 200,
    # Number of actors used to collect the rollouts
    "num_workers": 5,
    # Resource requirements for remote actors
    "worker_resources": {"num_cpus": None},
    # Dump TensorFlow timeline after this many SGD minibatches
    "full_trace_nth_sgd_batch": -1,
    # Whether to profile data loading
    "full_trace_data_load": False,
    # Outer loop iteration index when we drop into the TensorFlow debugger
    "tf_debug_iteration": -1,
    # If this is True, the TensorFlow debugger is invoked if an Inf or NaN
    # is detected
    "tf_debug_inf_or_nan": False,
    # If True, we write tensorflow logs and checkpoints
    "write_logs": True,
    # Arguments to pass to the env creator
    "env_config": {},
}

ENV_SEED = 123


class PPOAgent(Agent):
    _agent_name = "PPO"
    _allow_unknown_subkeys = ["model", "tf_session_args", "env_config",
                              "worker_resources"]
    _default_config = DEFAULT_CONFIG

    def _init(self):

        self.shared_model = (self.config["model"].get("custom_options", {}).
                             get("multiagent_shared_model", False))
        if self.shared_model:
            self.num_models = 1
        else:
            self.num_models = len(self.config["model"].get(
                "custom_options", {}).get("multiagent_obs_shapes", [1]))
        self.global_step = 0
        self.timesteps = 0
        self.kl_coeff = [self.config["kl_coeff"]] * self.num_models
        self.local_evaluator = PPOEvaluator(
            self.registry, self.env_creator, self.config, self.logdir, False)
        RemotePPOEvaluator = ray.remote(
            **self.config["worker_resources"])(PPOEvaluator)
        self.remote_evaluators = [
            RemotePPOEvaluator.remote(
                self.registry, self.env_creator, self.config, self.logdir,
                True)
            for _ in range(self.config["num_workers"])]

        self.w_policy = self.local_evaluator.get_weights()
        print(self.w_policy.shape)
        self.num_deltas = self.w_policy.size  # number of perturbation directions

        self.start_time = time.time()
        if self.config["write_logs"]:
            self.file_writer = tf.summary.FileWriter(
                self.logdir, self.local_evaluator.sess.graph)
        else:
            self.file_writer = None
        self.saver = tf.train.Saver(max_to_keep=None)

        # Create the actors for ARS setup - nskh
        print("Creating actors.")
        self.workers = [
            Worker.remote(
                self.registry, self.config, self.env_creator,
                ENV_SEED + 7 * i,
                evaluator=self.local_evaluator)
            for i in range(self.config["num_workers"])]

    def _train(self):
        agents = self.remote_evaluators
        config = self.config
        model = self.local_evaluator

        if (config["num_workers"] * config["min_steps_per_task"] >
                config["timesteps_per_batch"]):
            print(
                "WARNING: num_workers * min_steps_per_task > "
                "timesteps_per_batch. This means that the output of some "
                "tasks will be wasted. Consider decreasing "
                "min_steps_per_task or increasing timesteps_per_batch.")

        print("===> iteration", self.iteration)

        iter_start = time.time()
        weights = ray.put(model.get_weights())
        [a.set_weights.remote(weights) for a in agents]
        samples = collect_samples(agents, config, self.local_evaluator)

        def standardized(value):
            # Divide by the maximum of value.std() and 1e-4
            # to guard against the case where all values are equal
            return (value - value.mean()) / max(1e-4, value.std())

        samples.data["advantages"] = standardized(samples["advantages"])

        rollouts_end = time.time()

        # TODO(nskh) collect g_hats somehow
        print('Computing empirical gradient')
        g_hat, info_dict = self.aggregate_rollouts()

        print("Computing policy (iterations=" + str(config["num_sgd_iter"]) +
              ", stepsize=" + str(config["sgd_stepsize"]) + "):")
        names = [
            "iter", "total loss", "policy loss", "vf loss", "kl", "entropy"]
        print(("{:>15}" * len(names)).format(*names))
        samples.shuffle()
        shuffle_end = time.time()
        tuples_per_device = model.load_data(
            samples, self.iteration == 0 and config["full_trace_data_load"])
        load_end = time.time()
        rollouts_time = rollouts_end - iter_start
        shuffle_time = shuffle_end - rollouts_end
        load_time = load_end - shuffle_end
        sgd_time = 0
        kl = []
        for i in range(config["num_sgd_iter"]):
            sgd_start = time.time()
            batch_index = 0
            num_batches = (
                int(tuples_per_device) // int(model.per_device_batch_size))
            loss, policy_loss, vf_loss, kl, entropy = [], [], [], [], []
            permutation = np.random.permutation(num_batches)
            # Prepare to drop into the debugger
            if self.iteration == config["tf_debug_iteration"]:
                model.sess = tf_debug.LocalCLIDebugWrapperSession(model.sess)
            while batch_index < num_batches:
                full_trace = (
                    i == 0 and self.iteration == 0 and
                    batch_index == config["full_trace_nth_sgd_batch"])
                batch_loss, batch_policy_loss, batch_vf_loss, batch_kl, \
                    batch_entropy = model.run_sgd_minibatch(
                        permutation[batch_index] * model.per_device_batch_size,
                        self.kl_coeff, full_trace,
                        self.file_writer)
                loss.append(batch_loss)
                policy_loss.append(batch_policy_loss)
                vf_loss.append(batch_vf_loss)
                kl.append(batch_kl)
                entropy.append(batch_entropy)
                batch_index += 1
            loss = np.mean(loss)
            policy_loss = np.mean(policy_loss)
            vf_loss = np.mean(vf_loss)
            kl = np.mean(kl)
            entropy = np.mean(entropy)
            sgd_end = time.time()
            print(
                "{:>15}{:15.5e}{:15.5e}{:15.5e}{:15.5e}{:15.5e}".format(
                    i, loss, policy_loss, vf_loss, kl, entropy))

            values = []
            if i == config["num_sgd_iter"] - 1:
                metric_prefix = "ppo/sgd/final_iter/"
                values.append(tf.Summary.Value(
                    tag=metric_prefix + "kl_coeff",
                    simple_value=np.mean(self.kl_coeff)))
                values.extend([
                    tf.Summary.Value(
                        tag=metric_prefix + "mean_entropy",
                        simple_value=entropy),
                    tf.Summary.Value(
                        tag=metric_prefix + "mean_loss",
                        simple_value=loss),
                    tf.Summary.Value(
                        tag=metric_prefix + "mean_kl",
                        simple_value=kl)])
                if self.file_writer:
                    sgd_stats = tf.Summary(value=values)
                    self.file_writer.add_summary(sgd_stats, self.global_step)
            self.global_step += 1
            sgd_time += sgd_end - sgd_start

        # treat single-agent as a multi-agent system w/ one agent
        if not isinstance(kl, np.ndarray):
            kl = [kl]

        for i, kl_i in enumerate(kl):
            if kl_i > 2.0 * config["kl_target"]:
                self.kl_coeff[i] *= 1.5
            elif kl_i < 0.5 * config["kl_target"]:
                self.kl_coeff[i] *= 0.5

        info = {
            "kl_divergence": np.mean(kl),
            "kl_coefficient": np.mean(self.kl_coeff),
            "rollouts_time": rollouts_time,
            "shuffle_time": shuffle_time,
            "load_time": load_time,
            "sgd_time": sgd_time,
            "sample_throughput": len(samples["obs"]) / sgd_time
        }

        FilterManager.synchronize(
            self.local_evaluator.filters, self.remote_evaluators)
        res = self._fetch_metrics_from_remote_evaluators()
        res = res._replace(info=info)
        return res

    def _fetch_metrics_from_remote_evaluators(self):
        episode_rewards = []
        episode_lengths = []
        metric_lists = [a.get_completed_rollout_metrics.remote()
                        for a in self.remote_evaluators]
        for metrics in metric_lists:
            for episode in ray.get(metrics):
                episode_lengths.append(episode.episode_length)
                episode_rewards.append(episode.episode_reward)
        avg_reward = (
            np.mean(episode_rewards) if episode_rewards else float('nan'))
        avg_length = (
            np.mean(episode_lengths) if episode_lengths else float('nan'))
        timesteps = np.sum(episode_lengths) if episode_lengths else 0

        result = TrainingResult(
            episode_reward_mean=avg_reward,
            episode_len_mean=avg_length,
            timesteps_this_iter=timesteps)

        return result

    def _stop(self):
        # workaround for https://github.com/ray-project/ray/issues/1516
        for ev in self.remote_evaluators:
            ev.__ray_terminate__.remote(ev._ray_actor_id.id())

    def _save(self, checkpoint_dir):
        checkpoint_path = self.saver.save(
            self.local_evaluator.sess,
            os.path.join(checkpoint_dir, "checkpoint"),
            global_step=self.iteration)
        agent_state = ray.get(
            [a.save.remote() for a in self.remote_evaluators])
        extra_data = [
            self.local_evaluator.save(),
            self.global_step,
            self.kl_coeff,
            agent_state]
        pickle.dump(extra_data, open(checkpoint_path + ".extra_data", "wb"))
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.saver.restore(self.local_evaluator.sess, checkpoint_path)
        extra_data = pickle.load(open(checkpoint_path + ".extra_data", "rb"))
        self.local_evaluator.restore(extra_data[0])
        self.global_step = extra_data[1]
        self.kl_coeff = extra_data[2]
        ray.get([
            a.restore.remote(o)
                for (a, o) in zip(self.remote_evaluators, extra_data[3])])

    def compute_action(self, observation):
        observation = self.local_evaluator.obs_filter(
            observation, update=False)
        return self.local_evaluator.common_policy.compute(observation)[0]

    # FIXME(ev) should return the rewards and some other statistics
    def aggregate_rollouts(self, num_rollouts=None, evaluate=False):
        """
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts

        # TODO(nskh): figure out how to grab and set these weights
        # refresh weights
        flat_weights = self.local_evaluator.get_weights(flat=True)
        # how do you modify remote evaluators? do i need to?
        # for re in self.remote_evaluators:
        #     re.get_weights()

        # put policy weights in the object store
        policy_id = ray.put(flat_weights)

        t1 = time.time()

        # parallel generation of rollouts
        rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                     evaluate=evaluate)
                           for worker in self.workers]

        remainder_workers = self.workers[:(num_deltas % self.config["num_workers"])]
        # handle the remainder of num_delta/num_workers
        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                     evaluate=evaluate)
                           for worker in remainder_workers]

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
        deltas_idx = np.array(deltas_idx)  # probably unnecessary - nskh
        rollout_rewards = np.array(rollout_rewards, dtype=np.float64)

        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards

        t1 = time.time()

        # aggregate rollouts to form the gradient used to compute SGD step

        # reward_diff is vector of positive diff reward minus negative diff reward
        reward_diff = rollout_rewards[:, 0] - rollout_rewards[:, 1]

        deltas_tuple = (make_elementary_vector(idx, flat_weights.shape) for idx in deltas_idx)

        # this should be fine - nskh
        g_hat, count = utils.batched_weighted_sum(reward_diff, deltas_tuple,
                                                  batch_size=500)

        # TODO(nskh): why does this division occur?
        g_hat /= deltas_idx.size
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat, info_dict


# TODO(nskh): should workers actually be remote_evaluators? how hard would this be
@ray.remote
class Worker(object):
    """
    Object class for parallel rollout generation.
    """

    def __init__(self, registry, config, env_creator,
                 env_seed,
                 evaluator=None):

        # initialize OpenAI environment for each worker
        self.env = env_creator(config["env_config"])
        self.env.seed(env_seed)

        from ray.rllib import models
        self.preprocessor = models.ModelCatalog.get_preprocessor(
            registry, self.env)

        from ray.rllib import models
        self.preprocessor = models.ModelCatalog.get_preprocessor(
            registry, self.env)

        self.rollout_length = self.env.spec.max_episode_steps,
        self.sess = utils.make_session(single_threaded=True)
        if evaluator is not None:
            self.evaluator = evaluator
            self.policy = evaluator.common_policy
            # self.env = evaluator.env  # should be unnecessary

    def rollout(self, shift=0., rollout_length=None):
        """
        Performs one rollout of maximum length.
        At each time-step it subtracts shift from the reward.
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

    def do_rollouts(self, w_policy, shift=1, evaluate=False, sample=False):
        """
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, steps, deltas_idx = [], [], []

        # TODO(nskh) fix loop iteration number

        # TODO(nskh) verify policy shape
        num_weights = w_policy.size
        for i in range(num_weights):
            if evaluate:
                self.policy.set_weights(w_policy, flat=True)
                deltas_idx.append(-1)

                # for evaluation we do not shift the rewards (shift = 0)
                # and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                time_limit = self.env.spec.timestep_limit
                reward, r_steps = self.rollout(shift=0.,
                                               rollout_length=time_limit)
                rollout_rewards.append(reward)

            else:
                # TODO(nskh) verify policy shape
                delta = make_elementary_vector(i, w_policy.shape)
                deltas_idx.append(i)

                # compute reward and number of timesteps used
                # for positive perturbation rollout
                self.policy.set_weights(w_policy + delta, flat=True)
                if not sample:
                    pos_reward, pos_steps = self.rollout(shift=shift)
                else:
                    # TODO(nskh) average reward over rollouts if stochastic
                    pos_reward, pos_steps = 0, 0
                    pass

                # compute reward and number of timesteps used f
                # or negative pertubation rollout
                self.policy.set_weights(w_policy - delta, flat=True)
                if not sample:
                    neg_reward, neg_steps = self.rollout(shift=shift)
                else:
                    # TODO(nskh) average reward over rollouts if stochastic
                    neg_reward, neg_steps = 0, 0
                    pass
                steps += [pos_steps, neg_steps]

                rollout_rewards.append([pos_reward, neg_reward])

        return {'deltas_idx': deltas_idx,
                'rollout_rewards': rollout_rewards,
                "steps": steps}


def make_elementary_vector(idx, shape, step_size=1.0):
    vec = np.zeros(shape)
    # TODO(nskh) verify size of policy shape
    vec[idx] = 1.0*step_size  # TODO(nskh) modularize perturbation size
    return vec
