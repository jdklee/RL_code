from collections import deque

import numpy as np
import torch
import gym
import os
from general import get_logger, Progbar, export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, device, np2torch
from policy import CategoricalPolicy, GaussianPolicy

class PolicyGradient(object):
    """
    Class for implementing a general policy gradient algorithm that uses baseline to approximate value function.
    Any further, more recent algorithms like PPO or TPRO can be implemented using this setup.
    In essence, the baseline is the Critic Network of any Actor-Critic based policy gradient algorithm.
    """

    def __init__(self, env, config, seed, PPO=True, logger=None, normalize_advantage_flag=True,
                 max_ep_len=500, learning_rate=3e-2):
        # directory for training outputs

        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyperparameters
        self.seed = seed
        self.normalize_advantage_flag=normalize_advantage_flag
        self.max_ep_len = max_ep_len
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env
        self.env.seed(self.seed)

        # discrete vs continuous action space
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = (
            self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        )

        self.lr = learning_rate
        self.num_batches = 100  # number of batches trained on
        self.batch_size = 50000  # number of steps used to compute each policy update
        self.max_ep_len = 1000  # maximum episode length
        self.learning_rate = 3e-2
        self.gamma = 0.9  # the discount factor

        self.n_layers = 2
        self.layer_size = 64

        self.init_policy()
        self.PPO=PPO
        self.baseline_network = BaselineNetwork(env, n_layers=self.n_layers, layer_size=self.layer_size, learning_rate=self.lr)
        self.baseline_network.action_dim = self.action_dim
        self.baseline_network.observation_dim = self.observation_dim
        if PPO:
            self.n_updates_per_iteration=5
            self.clip=0.2




    def init_policy(self):

        net = build_mlp(input_size=self.observation_dim, output_size=self.action_dim,
                        size=self.config.layer_size, n_layers=self.config.n_layers)

        if self.discrete:
            self.policy = CategoricalPolicy(net)
        else:
            self.policy = GaussianPolicy(net, action_dim=self.action_dim)
        self.optimizer = torch.optim.Adam(self.policy.network.parameters(), lr=self.lr)

    def init_averages(self):
        self.avg_reward = 0.0
        self.max_reward = 0.0
        self.std_reward = 0.0
        self.eval_reward = 0.0

    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.
        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        pass

    def sample_path(self, env, num_episodes=None):
        """
        Sample paths (trajectories) from the environment.

        Args:
            num_episodes: the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            env: open AI Gym envinronment

        Returns:
            paths: a list of paths. Each path in paths is a dictionary with
                path["observation"] a numpy array of ordered observations in the path
                path["actions"] a numpy array of the corresponding actions in the path
                path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards: the sum of all rewards encountered during this "path"
        """
        episode = 0
        episode_rewards = []
        paths = []
        t = 0

        while num_episodes or t < self.config.batch_size:
            state = env.reset()
            states, actions, rewards, values, log_probs = [], [], [], [], []
            episode_reward = 0

            for step in range(self.max_ep_len):
                states.append(state)
                action, log_prob = self.policy.act(states[-1][None])[0]
                log_probs.append(log_prob)
                state, reward, done, info = env.step(action)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if done or step == self.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    break

            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
                "log_prob":np.array(log_probs)
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards

    def get_returns(self, paths):
        """
        Calculate the returns G_t for each timestep

        Args:
            paths: recorded sample paths. See sample_path() for details.

        Return:
            returns: return G_t for each timestep

        After acting in the environment, we record the observations, actions, and
        rewards. To get the advantages that we need for the policy update, we have
        to convert the rewards into returns, G_t, which are themselves an estimate
        of Q^π (s_t, a_t):

           G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T

        where T is the last timestep of the episode.

        Note that here we are creating a list of returns for each path
        """

        all_returns = []
        for path in paths:
            rewards = path["reward"]
            returns = np.zeros_like(rewards)
            returns[-1] = rewards[-1]

            for i in range(len(rewards) - 1):
                returns[i] = rewards[i] + sum([self.config.gamma ** j * rewards[j] for j in range(i + 1, len(rewards))])

            all_returns.append(returns)

        returns = np.concatenate(all_returns)

        return returns

    def normalize_advantage(self, advantages):
        """
        Args:
            advantages: np.array of shape [batch size]
        Returns:
            normalized_advantages: np.array of shape [batch size]

        Note:
        This function is called only if self.config.normalize_advantage is True.
        """
        #######################################################
        #########   YOUR CODE HERE - 1-2 lines.    ############
        normalized_advantages = advantages / np.linalg.norm(advantages)

        #######################################################
        #########          END YOUR CODE.          ############
        return normalized_advantages

    def calculate_advantage(self, returns, observations):
        """
        Calculates the advantage for each of the observations
        Args:
            returns: np.array of shape [batch size]
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]
        """
        if self.config.use_baseline:
            # override the behavior of advantage by subtracting baseline
            advantages = self.baseline_network.calculate_advantage(
                returns, observations
            )
        else:
            advantages = returns

        if self.normalize_advantage_flag:
            advantages = self.normalize_advantage(advantages)

        return advantages


    def update_policy(self, loss):
        """
        Perform one update on the policy using the provided loss
        """
        self.loss=loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


    def train(self):
        """
        Performs training
        """
        last_record = 0


        self.init_averages()
        all_total_rewards = (
            []
        )  # the returns of all episodes samples for training purposes
        averaged_total_rewards = []  # the returns for each iteration

        for t in range(self.config.num_batches):

            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.env)
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            batch_log = np.concatenate([path["log_prob"] for path in paths])
            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)

            # advantage will depend on the baseline implementation
            advantages = self.calculate_advantage(returns, observations)

            observations = np2torch(observations)
            actions = np2torch(actions)
            advantages = np2torch(advantages)
            returns = np2torch(returns)

            # run training operations
            for _ in range(self.n_updates_per_iteration):
                #Update policy
                dist = self.policy.action_distribution(observations)
                log_probabilties = dist.log_prob(actions)
                if not self.PPO:
                    policy_loss = -torch.sum(log_probabilties * advantages)
                else:
                    ratios = torch.exp(log_probabilties - batch_log)
                    surrogate_loss_1 = ratios * advantages
                    surrogate_loss_2 = torch.clamp(ratios, 1-self.clip, 1+self.clip)*advantages
                    policy_loss=(-torch.min(surrogate_loss_2, surrogate_loss_1)).mean()

                self.update_policy(policy_loss)

                # update value
                value_loss = torch.nn.functional.mse_loss(returns, self.baseline_network.forward(observations))
                self.baseline_network.update_baseline(value_loss)

            # logging
            if t % self.config.summary_freq == 0:
                self.update_averages(total_rewards, all_total_rewards)
                self.record_summary(t)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
                avg_reward, sigma_reward
            )
            averaged_total_rewards.append(avg_reward)
            self.logger.info(msg)

            if self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        self.logger.info("- Training done.")
        np.save(self.config.scores_output, averaged_total_rewards)
        export_plot(
            averaged_total_rewards,
            "Score",
            self.config.env_name,
            self.config.plot_output,
        )


    def evaluate(self, env=None, num_episodes=1):
        """
        Evaluates the return for num_episodes episodes.
        Not used right now, all evaluation statistics are computed during training
        episodes.
        """
        if env == None:
            env = self.env
        paths, rewards = self.sample_path(env, num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)
        return avg_reward

    def record(self):
        """
        Recreate an env and record a video for one episode
        """
        env = gym.make(self.config.env_name)
        env.seed(self.seed)
        env = gym.wrappers.Monitor(
            env, self.config.record_path, video_callable=lambda x: True, resume=True
        )
        self.evaluate(env, 1)

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # record one game at the beginning
        if self.config.record:
            self.record()
        # model
        self.train()
        # record one game at the end
        if self.config.record:
            self.record()
