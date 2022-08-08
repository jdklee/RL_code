from collections import deque

import numpy as np
import torch
import gym
import numpy as np
import torch
import gym
import torch.nn as nn
import os
from general import get_logger, Progbar, export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, device, np2torch
from policy import CategoricalPolicy, GaussianPolicy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np

class Critic(nn.Module):
    """
    Class for implementing Critic network that approximates the value of each state
    """

    def __init__(self, env, config):

        super().__init__()
        self.config = config
        self.env = env
        self.baseline = None
        self.lr = self.config.learning_rate


        #######################################################
        #########   YOUR CODE HERE - 2-8 lines.   #############
        self.observation_dim = self.env.observation_space.shape[0]
        self.network =build_mlp(input_size=self.observation_dim, size=self.config.layer_size, n_layers=self.config.n_layers,
                  output_size=1)
        self.optimizer = torch.optim.Adam(self.network.parameters() ,lr=self.lr)

        #######################################################
        #########          END YOUR CODE.          ############

    def forward(self, observations):
        output=self.network(observations).squeeze()
        assert output.ndim == 1
        return output

    def calculate_advantage(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]

        TODO:
        Evaluate the baseline and use the result to compute the advantages.
        Put the advantages in a variable called "advantages" (which will be
        returned).

        Note:
        The arguments and return value are numpy arrays. The np2torch function
        converts numpy arrays to torch tensors. You will have to convert the
        network output back to numpy, which can be done via the numpy() method.
        """
        observations = np2torch(observations)

        values=self.forward(observations)

        advantages = returns - values.detach().numpy()

        return advantages

    def update_value(self, reward, new_observation, old_observation, I, done):
        """
        Args:
            returns: np.array of shape [batch size], containing all discounted
                future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]

        TODO:
        Compute the loss (MSE), backpropagate, and step self.optimizer.
        You may (though not necessary) find it useful to do perform these steps
        more than one once, since this method is only called once per policy update.
        If you want to use mini-batch SGD, we have provided a helper function
        called batch_iterator (implemented in general.py).
        """
        reward = np2torch(reward)
        new_observation = np2torch(new_observation)
        old_observation = np2torch(old_observation)

        #######################################################
        #########   YOUR CODE HERE - 4-10 lines.  #############
        new_state_val=self.network(new_observation)
        old_state_val=self.network(old_observation)
        if done:
            expected_returns = reward
        else:
            expected_returns = reward*self.config.gamma+new_state_val
        self.loss=torch.nn.functional.mse_loss(expected_returns, old_state_val) * I


        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

class Actor(nn.Module):
    """
    Class for implementing Critic network that approximates the value of each state
    """

    def __init__(self, env, config):

        super().__init__()
        self.config = config
        self.env = env
        self.baseline = None
        self.lr = self.config.learning_rate


        #######################################################
        #########   YOUR CODE HERE - 2-8 lines.   #############
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = (
            self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        )
        self.network =build_mlp(input_size=self.observation_dim, size=self.config.layer_size, n_layers=self.config.n_layers,
                  output_size=self.action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters() ,lr=self.lr)

        if self.discrete:
            self.policy = CategoricalPolicy(self.network)
        else:
            self.policy = GaussianPolicy(self.network, action_dim=self.action_dim)


        #######################################################
        #########          END YOUR CODE.          ############



    def forward(self, observations):
        output=self.network(observations).squeeze()
        assert output.ndim == 1
        return output

    def calculate_advantage(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]

        TODO:
        Evaluate the baseline and use the result to compute the advantages.
        Put the advantages in a variable called "advantages" (which will be
        returned).

        Note:
        The arguments and return value are numpy arrays. The np2torch function
        converts numpy arrays to torch tensors. You will have to convert the
        network output back to numpy, which can be done via the numpy() method.
        """
        observations = np2torch(observations)

        values=self.forward(observations)

        advantages = returns - values.detach().numpy()

        return advantages


    def update_policy(self, reward, new_state, old_state, action, I, done):
        new_state = np2torch(new_state)
        old_state = np2torch(old_state)
        reward = np2torch(reward)
        action = np2torch(action)
        #######################################################
        #########   YOUR CODE HERE - 5-7 lines.    ############
        dist = self.policy.action_distribution(new_state)
        log_probabilties = dist.log_prob(action)
        if done:
            advantage = reward - self.Critic(old_state)
        else:
            advantage = reward + self.config.gamma * self.Critic(new_state) - self.Critic(old_state)
        self.loss = -torch.sum(log_probabilties * advantage) * I
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def train_actor_critic(self, num_episodes):
        """
        Performs training

        """

        scores=[]
        recent_scores=deque(maxlen=100)


        for t in range(self.config.num_batches):
            env = self.env
            state = env.reset()
            done=False
            I = 1
            score=0

            for step in range(self.config.max_ep_len):
                action = self.policy.act(state)[0]
                new_state, reward, done, info = env.step(action)
                # value = self.Critic(new_state)
                # Q_SA = reward + self.config.gamma * self.Critic(new_state) if done else reward
                # td_target = Q_SA -self.Critic(state)
                score+=reward

                # run training operations
                self.Critic.update_value(reward=reward, new_observation=new_state, old_observation=state, I=I, done=done)

                self.update_policy(reward=reward, new_state=new_state, old_state=state, I=I, action=action, done=done)
                if done:
                    break

                I *= self.config.gamma
                state=new_state

            scores.append(score)
            recent_scores.append(score)

    def show_results(self, scores, recent_scores, sklearn=None):


        sns.set()

        plt.plot(scores)
        plt.ylabel('score')
        plt.xlabel('episodes')
        plt.title('Training score of CartPole Actor-Critic TD(0)')

        reg = LinearRegression().fit(np.arange(len(scores)).reshape(-1, 1), np.array(scores).reshape(-1, 1))
        y_pred = reg.predict(np.arange(len(scores)).reshape(-1, 1))
        plt.plot(y_pred)
        plt.show()

    def show_last_policy_results(self):
        done = False
        state = self.env.reset()
        scores = []

        for _ in range(50):
            score = 0
            while not done:
                # env.render()
                action = self.policy.act(state)
                new_state, reward, done, info = self.env.step(action)
                score += reward
                state = new_state
            scores.append(score)
        self.env.close()
        print(np.array(scores).mean())