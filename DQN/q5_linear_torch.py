import copy
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from utils.test_env import EnvTest
from core.deep_q_learning_torch import DQN
from q4_schedule import LinearExploration, LinearSchedule

from configs.q5_linear import config


class Linear(DQN):
    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a linear layer with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?
        """
        # this information might be useful
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n

        self.q_network=torch.nn.Linear(img_height*img_width*n_channels*self.config.state_history, num_actions)
        self.target_network = torch.nn.Linear(img_height*img_width*n_channels*self.config.state_history, num_actions)



    def get_q_values(self, state, network='q_network'):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)
            """

        if network=="q_network":
            out= self.q_network(torch.flatten(state, start_dim=1))
        else:
            out= self.target_network(torch.flatten(state, start_dim=1))

        return out


    def update_target(self):
        """
        update_target_op will be called periodically
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights.

        Periodically, we need to update all the weights of the Q network
        and assign them with the values from the regular network.
        """

        torch.save(self.q_network, "q_network_weights")
        self.target_network=torch.load("q_network_weights")



    def calc_loss(self, q_values : Tensor, target_q_values : Tensor,
                    actions : Tensor, rewards: Tensor, done_mask: Tensor) -> Tensor:
        """
        Calculate the MSE loss of this step.
        The loss for an example is defined as:
            Q_samp(s) = r if done
                        = r + gamma * max_a' Q_target(s', a')
            loss = (Q_samp(s) - Q(s, a))^2

        Args:
            q_values: (torch tensor) shape = (batch_size, num_actions)
                The Q-values that your current network estimates (i.e. Q(s, a') for all a')
            target_q_values: (torch tensor) shape = (batch_size, num_actions)
                The Target Q-values that your target network estimates (i.e. (i.e. Q_target(s', a') for all a')
            actions: (torch tensor) shape = (batch_size,)
                The actions that you actually took at each step (i.e. a)
            rewards: (torch tensor) shape = (batch_size,)
                The rewards that you actually got at each step (i.e. r)
            done_mask: (torch tensor) shape = (batch_size,)
                A boolean mask of examples where we reached the terminal state
        """
        # you may need this variable
        num_actions = self.env.action_space.n
        actions_taken=torch.nn.functional.one_hot(actions.to(torch.int64), num_classes=num_actions)
        q_val_after_done=~done_mask[:,None]*target_q_values

        max_q_target, max_q_target_idx = torch.max(q_val_after_done, dim=1)
        gamma = self.config.gamma
        q_samp = rewards + gamma * max_q_target
        q_sa=torch.sum(q_values*actions_taken, dim=1)

        loss=torch.nn.functional.mse_loss(q_samp, q_sa)
        return loss


    def add_optimizer(self, lr=0.00001):
        """
        Set self.optimizer to be an Adam optimizer optimizing only the self.q_network
        parameters
        """
        self.optimizer=torch.optim.Adam(self.q_network.parameters(), lr=lr)




if __name__ == '__main__':
    env = EnvTest((5, 5, 1))
    # log=logging.getLogger('matplotlib.font_manager').disabled = True

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
