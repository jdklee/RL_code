import numpy as np
import torch
import gym
import torch.nn as nn
from network_utils import build_mlp, device, np2torch
"""self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = (
            self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        )"""

class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network to approximate state-value function
    """

    def __init__(self, config, env, n_layers, layer_size, learning_rate):
        """
        TODO:
        Create self.network using build_mlp, and create self.optimizer to
        optimize its parameters.
        You should find some values in the config, such as the number of layers,
        the size of the layers, and the learning rate.
        """
        super().__init__()
        self.config = config
        self.env = env
        self.baseline = None
        self.lr = self.learning_rate
        self.layer_size = layer_size
        self.n_layers=n_layers



        #######################################################
        #########   YOUR CODE HERE - 2-8 lines.   #############
        self.observation_dim = self.env.observation_space.shape[0]
        self.network =build_mlp(input_size=self.observation_dim, size=self.layer_size, n_layers=self.n_layers,
                  output_size=1)
        self.optimizer = torch.optim.Adam(self.network.parameters() ,lr=self.lr)

        #######################################################
        #########          END YOUR CODE.          ############

    def forward(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]

        TODO:
        Run the network forward and then squeeze the result so that it's
        1-dimensional. Put the squeezed result in a variable called "output"
        (which will be returned).

        Note:
        A nn.Module's forward method will be invoked if you
        call it like a function, e.g. self(x) will call self.forward(x).
        When implementing other methods, you should use this instead of
        directly referencing the network (so that the shape is correct).
        """
        #######################################################
        #########   YOUR CODE HERE - 1 lines.     #############
        output=self.network(observations).squeeze()
        # print("output of baseline nw:",output.shape)

        #######################################################
        #########          END YOUR CODE.          ############
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
        #######################################################
        #########   YOUR CODE HERE - 1-4 lines.   ############
        baseline=self.forward(observations)

        advantages = returns - baseline.detach().numpy()


        #######################################################
        #########          END YOUR CODE.          ############
        return advantages

    def update_baseline(self, loss):
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
        # returns = np2torch(returns)
        # observations = np2torch(observations)
        #######################################################
        #########   YOUR CODE HERE - 4-10 lines.  #############
        # print("return shape=",returns.shape)
        # print("obs shape=",observations.shape)
        # print("acself.action_dim)
        # print("forward pass shape:",self.network(observations).squeeze().shape)
        # print(self.forward(observations))
        # print(returns)
        # self.loss=torch.nn.functional.mse_loss(returns, self.network(observations).squeeze())
        self.loss=loss


        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        #######################################################
        #########          END YOUR CODE.          ############
