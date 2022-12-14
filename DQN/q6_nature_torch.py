import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

from utils.general import get_logger
from utils.test_env import EnvTest
from q4_schedule import LinearExploration, LinearSchedule
from q5_linear_torch import Linear


from configs.q6_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    Model configuration can be found in the Methods section of the above paper.
    """

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?

        To simplify, we specify the paddings as:
            (stride - 1) * img_height - stride + filter_size) // 2

The exact architecture, shown schematically in Fig. 1, is as follows. The input to the neural network consists of an
84 x 84 x 4 image produced by the preprocessing map phi_.
The first hidden layer convolves 32 filters of 8 x 8 with stride 4 with the input image and applies a rectifier nonlinearity
The second hidden layer convolves 64 filters of 4 x 4 with stride 2, again followed by a rectifier nonlinearity.
third convolutional layer that convolves 64 filters of 3 x 3 with stride 1 followed by a rectifier.
The final hidden layer is fully-connected and consists of 512 rectifier units.
The output layer is a fully-connected linear layer with a single output for each valid action.
The number of valid actions varied between 4 and 18 on the games we considered.
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n

        self.q_network = nn.Sequential(
            nn.Conv2d(in_channels=n_channels*self.config.state_history,
                      out_channels=32,
                      kernel_size=8, stride=4,
                      padding=(3*img_height-4+8)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4, stride=2,
                      padding=(1 * img_height - 2 + 4) // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3, stride=1,
                      padding=(0 * img_height - 1 + 3) // 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(img_height*img_width*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        self.target_network=nn.Sequential(
            nn.Conv2d(in_channels=n_channels*self.config.state_history,
                      out_channels=32,
                      kernel_size=8, stride=4,
                      padding=(3*img_height-4+8)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4, stride=2,
                      padding=(1 * img_height - 2 + 4) // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3, stride=1,
                      padding=(0 * img_height - 1 + 3) // 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(img_height*img_width*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def get_q_values(self, state, network):
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

        x=torch.stack([state[i].T for i in range(len(state))])
        if network=="q_network":
            out=self.q_network(x)
        else:
            out=self.target_network(x)

        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
