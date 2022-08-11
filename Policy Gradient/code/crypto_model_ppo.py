import math
import os
from collections import deque
from general import get_logger, Progbar, export_plot
import numpy as np
import torch
import torch.nn as nn
from baseline_network import *
from network_utils import *

class Actor(nn.Module):
    def __init__(self, env, input_shape, action_space, lr, gamma, n_layers=10, size=512, batch_size=64, logger=None, output_path="output"):
        super().__init__()
        self.env=env
        self.observation_size=500
        input_tensor=torch.randn(input_shape)
        self.action_space=action_space
        self.max_ep_len=500
        self.clip=0.2
        self.lr=lr
        self.gamma=gamma
        self.n_updates_per_iteration=5
        size=512
        n_layers=10
        output_size=len(action_space)
        net = [nn.Linear(math.prod(input_shape), size), nn.ReLU()]

        for _ in range(n_layers):
            net.append(nn.Linear(size, size))
            net.append(nn.ReLU())

        net.append(nn.Linear(size, output_size))
        self.network=nn.Sequential(*net)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.batch_size=batch_size
        self.logger = logger
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_path=output_path
        self.plot_output = self.output_path + "/scores.png"
        self.scores_output = self.output_path + "/scores.npy"
        self.summary_freq = 1
        # store hyperparameters
        self.log_path = self.output_path + "/log.txt"
        if logger is None:
            self.logger = get_logger(self.log_path)

        self.Critic=Critic(env=self.env, n_layers=n_layers, layer_size=size, learning_rate=self.lr)

    def forward(self, state):
        output = self.network(state)
        return output

    def sample_path(self, env, num_episodes=None):
        episode = 0
        episode_rewards = []
        paths = []
        t = 0
        print("sampling path")

        while t < num_episodes:
            state = env.reset(env_steps_size=500)
            env.render()
            states, next_states,actions, rewards, values, log_probs, predictions =[], [], [], [], [], [], []
            episode_reward = 0
            # print(t)

            for step in range(self.max_ep_len):
                states.append(state)
                action, log_prob, prediction = self.act(state)
                predictions.append(prediction)
                log_probs.append(log_prob)
                state, reward, done,_ = env.step(action)
                next_states.append(state)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if done or step == self.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.batch_size:
                    break

            path = {
                "observation": np.array(states),
                "next_state":np.array(next_states),
                "reward": np.array(rewards),
                "action": np.array(actions),
                "log_prob":np.array(log_probs),
                "prediction":np.array(predictions),
                "next_states":np.array(next_states)
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards

    def get_returns(self, paths):
        all_returns = []
        for path in paths:
            rewards = path["reward"]
            returns = np.zeros_like(rewards)
            returns[-1] = rewards[-1]

            for i in range(len(rewards) - 1):
                returns[i] = rewards[i] + sum([self.gamma ** j * rewards[j] for j in range(i + 1, len(rewards))])

            all_returns.append(returns)

        returns = np.concatenate(all_returns)

        return returns
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
    def train(self, visualize, train_episodes=50, training_batch_size=500):
        # self.env.create_writer()
        all_total_rewards = []
        average_total_rewards=deque(maxlen=100)
        # total_avg=
        best_avg=0
        print("start training")
        for ep in range(train_episodes):
            # state = self.env.reset(env_steps_size=500)
            # self.env.render(visualize)
            # episode_rewards=[]
            # for step in range(self.max_ep_len):
            #     # print(step," out of ", self.max_ep_len)
            #     action, log_prob, prediction = self.act(state)
            #     new_state, reward, done,_ = self.env.step(action)
            #     all_total_rewards.append(reward)
            #     episode_rewards.append(reward)
            #     advantages = reward + abs((done-1)) * self.gamma * self.Critic.forward(new_state.flatten()) - self.Critic.forward(state.flatten())
            #     # print("updating policy")
            #
            #     # Update policy
            #     logits = self.network.forward(np2torch(state.flatten()))
            #     distribution = torch.distributions.Categorical(logits=logits)
            #     log_probabilties = distribution.log_prob(np2torch(action)).detach().numpy()
            #
            #     ratios = np.exp(log_probabilties - log_prob)
            #     surrogate_loss_1 = ratios * advantages
            #     surrogate_loss_2 = np.clip(ratios, 1 - self.clip, 1 + self.clip) * advantages
            #     policy_loss = (-torch.min(surrogate_loss_2, surrogate_loss_1)).mean()
            #
            #     self.update_policy(policy_loss)
            #
            #     # update value
            #     value_loss = torch.nn.functional.mse_loss(reward+abs((done-1)) * self.gamma * self.Critic.forward(new_state.flatten()),
            #                                               self.Critic.forward(state.flatten()))
            #     self.Critic.update_baseline(value_loss)

            paths, episode_rewards = self.sample_path(self.env, num_episodes=training_batch_size)
            self.env.render(visualize)
            all_total_rewards.extend(episode_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            next_states = np.concatenate([path["next_states"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            batch_log = np.concatenate([path["log_prob"] for path in paths])
            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)
            print(observations.shape)
            # t=[]
            # idxes=[]
            # for idx,x in enumerate(observations):
            #     if x.shape==(100,5):
            #         idxes.append(idx)
            #     t.append(x.shape)
            # # print(idxes)
            # print(list(set(t)))
            # values=np.zeros(len(observations))
            # for idx, i in enumerate(observations):
            #     i=i.flatten()
            #     values[idx]=self.Critic.forward(i)
            advantages=np2torch(returns)-self.Critic.forward(observations)
            advantages=advantages.detach().numpy()
            #Normalize advantages
            advantages = advantages / np.linalg.norm(advantages)
            observations = np2torch(observations)
            actions = np2torch(actions)
            advantages = np2torch(advantages)
            returns = np2torch(returns)
            for _ in range(self.n_updates_per_iteration):
                print("updating policy")

                # Update policy
                # logits=torch.zeros(len(observations))
                # for idx, i in enumerate(observations):
                #     i = i.flatten()
                #     i=self.network(i)
                #     print(i)
                #     logits[idx]=i
                # logits=torch.tensor(logits)
                # print(logits)
                logits=self.forward(observations)

                distribution = torch.distributions.Categorical(logits=logits)
                # dist = self.policy.action_distribution(observations)
                log_probabilties = distribution.log_prob(actions)
                ratios=log_probabilties-np2torch(batch_log)

                ratios = torch.exp(ratios)
                surrogate_loss_1 = ratios * advantages
                surrogate_loss_2 = torch.clamp(ratios, 1-self.clip, 1+self.clip)*advantages
                policy_loss=(-torch.min(surrogate_loss_2, surrogate_loss_1)).mean()

                self.update_policy(policy_loss)

                # update value
                value_loss = torch.nn.functional.mse_loss(returns, self.Critic.forward(observations))
                self.Critic.update_baseline(value_loss)

            if ep % self.summary_freq == 0:
                self.update_averages(episode_rewards, all_total_rewards)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(episode_rewards)
            sigma_reward = np.sqrt(np.var(episode_rewards) / len(episode_rewards))
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
                avg_reward, sigma_reward
            )
            average_total_rewards.append(avg_reward)
            self.logger.info(msg)
            avg=np.average(average_total_rewards)
            print("net worth {} {:.2f} {:.2f} {}".format(ep, self.env.net_worth, avg, self.env.episode_orders))
            # if ep > len(average_total_rewards):
            if best_avg < avg:
                best_avg = avg
                print("Saving model")
                torch.save(self.network, "Crypto_trader_Actor.h5")
                torch.save(self.Critic.network, "Crypto_trader_Critic.h5")

        self.logger.info("- Training done.")
        np.save(self.scores_output, average_total_rewards)
        export_plot(
            average_total_rewards,
            "Score",
            self.env.env_name,
            self.plot_output,
        )
    def act(self, state):
        # print(np.expand_dims(state, axis=0).shape)
        state=np2torch(state)
        state=state.flatten()
        prediction = self.network(state)
        # print(prediction)
        dist = torch.distributions.Categorical(logits=prediction)
        action= dist.sample()
        log_proba = dist.log_prob(action)


        return action.detach().numpy(), log_proba.detach().numpy(), prediction.detach().numpy()

    def update_policy(self, loss):
        """
        Perform one update on the policy using the provided loss
        """
        self.loss=loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def test_agent(self, visualize=True, test_episodes=10):
        # env.Actor=torch.load("Crypto_trader_Actor.h5")
        # env.Actor.Critic=torch.load("Crypto_trader_Critic.h5")
        average_net_worth = 0
        for episode in range(test_episodes):
            state = self.env.reset()
            while True:
                self.env.render(visualize)
                action, log_proba, prediction = self.act(state)
                state, reward, done, _ = self.env.step(action)
                if self.env.current_step == self.env.end_step:
                    average_net_worth += self.env.net_worth
                    print("net_worth:", episode, self.env.net_worth, self.env.episode_orders)
                    break

        print("average {} episodes agent net_worth: {}".format(test_episodes, average_net_worth / test_episodes))

class Critic(nn.Module):
    def __init__(self, env, n_layers, layer_size, learning_rate):
        super().__init__()
        self.env = env
        self.baseline = None
        self.lr = learning_rate
        self.layer_size = layer_size
        self.n_layers=n_layers
        size=512
        n_layers=10
        output_size=1
        input_shape=self.env.state_shape
        net = [nn.Linear(math.prod(input_shape), size), nn.ReLU()]

        for _ in range(n_layers):
            net.append(nn.Linear(size, size))
            net.append(nn.ReLU())

        net.append(nn.Linear(size, output_size))
        self.network=nn.Sequential(*net)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, observations):
        try:
            observations=np2torch(observations)
        except:
            pass
        output=self.network(observations).squeeze()
        return output

    def update_baseline(self, loss):
        self.loss=loss
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


