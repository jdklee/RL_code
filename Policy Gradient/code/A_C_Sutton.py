import numpy as np
import torch
import pytorch_lightning as pl
# from pytorch_lightning import LightningModule, Trainer
# from pytorch_lightning.callbacks.progress import TQDMProgressBar
# from pytorch_lightning.loggers import CSVLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x

class critic(torch.nn.Module):
    def __init__(self, gamma, learning_rate, net_type, state_space, size, n_layers):
        super().__init__()
        self.lr = learning_rate
        self.net_type = net_type
        self.state_space, self.size, self.n_layers = state_space, size, n_layers
        if net_type == 'linear':
            # Flatten state vector for linear layer
            state_space = np.multiply(state_space)
            net = [torch.nn.Flatten(),
                   torch.nn.Linear(state_space, size)]
            for n_layer in n_layers:
                net.append(torch.nn.Linear(size, size))
                net.append(torch.nn.ReLU())
            net.append(torch.nn.Linear(size, self.action_space))
            self.network = torch.nn.Sequential(*net)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, state):
        state = torch.from_numpy(state)
        value = self.network(state)
        return value

    def update_policy(self, state, new_state, reward):

        # self.loss=torch.nn.CrossEntropyLoss(reward+self.gamma*self.critic(new_state), self.critic(state))
        # self.loss = torch.nn.functional.mse_loss(reward + self.gamma * self.critic(new_state), self.critic(state))
        self.loss=reward + self.gamma * self.critic(new_state) - self.critic(state)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

class actor(torch.nn.Module):
        def __init__(self, gamma, actor_learning_rate, critic_learning_rate,
                     net_type, state_space, action_space, size, n_layers, ppo=False,
                     discrete_action=False, discrete_state=False):
            super().__init__()
            self.lr=actor_learning_rate
            self.critic_lr=critic_learning_rate
            self.net_type=net_type
            self.ppo=ppo
            self.discrete_action=discrete_action
            self.discrete_state=discrete_state
            self.action_dim = (
                self.env.action_space.n if self.discrete_action else self.env.action_space.shape[0]
            )
            self.state_space, self.action_space, self.size, self.n_layers=state_space, action_space, size, n_layers
            if net_type=='linear':
                #Flatten state vector for linear layer
                state_space=np.multiply(state_space)
                net=[torch.nn.Flatten(),
                     torch.nn.Linear(state_space, size)]
                for n_layer in n_layers:
                    net.append(torch.nn.Linear(size, size))
                    net.append(torch.nn.ReLU())
                net.append(torch.nn.Linear(size, self.action_space))
                self.network=torch.nn.Sequential(*net)
            self.optimizer=torch.optim.Adam(self.network.parameters(), lr=self.lr)
            self.critic=critic(gamma=gamma, learning_rate=critic_learning_rate,
                               net_type=net_type, state_space=state_space,
                               size=size, n_layers=n_layers)

        def forward(self, state):
            state=torch.from_numpy(state)
            logits=self.network(state)
            return logits

        def get_action(self,state):
            logits=self.forward(state)

            if not self.discrete_action:
                distribution = torch.distributions.Categorical(logits=logits)
                sampled_action = distribution.sample()
                log_proba = distribution.log_prob(sampled_action).detach().numpy()
            else: #Gaussian if continuous action
                log_std = torch.nn.Parameter(torch.zeros(size=[self.action_space]))
                std = np.exp(log_std.detach().numpy())
                std = np2torch(std)
                distribution = torch.distributions.MultivariateNormal(loc=logits,
                                                              scale_tril=torch.diag(std))
                sampled_action = distribution.sample()
                log_proba = distribution.log_prob(sampled_action).detach().numpy()

            return sampled_action, log_proba

        def update_policy_ppo(self, paths):
            pass
            #TODO: implement collect_path to get states, rewards, advantages using current value function

        def update_policy(self, state, new_state, reward, logproba):
            if not self.ppo:
                #self.loss=torch.nn.CrossEntropyLoss(reward+self.gamma*self.critic(new_state), self.critic(state))
                self.loss=reward+self.gamma*self.critic(new_state) - self.critic(state)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()




class Actor_critic_agent:
    def __init__(self, env,  alpha_w, alpha_theta, alpha_r, critic, actor,
                 eligibility_trace=False, ppo=False):

        self.alpha_w=alpha_w
        self.alpha_theta=alpha_theta
        self.eligibility_trace = eligibility_trace
        self.env=env
        self.start_state=env.reset()
        self.critic=critic
        self.actor=actor
        self.ppo=ppo
    def collect_path(self, batch_size):
        pass
        # TODO: implement collect_path to get states, rewards, advantages using current value function

    def train(self, train_episodes, epochs):
        # self.log = {"value_loss": [], "policy_loss": [], "score": [], "episodic_score": [],
        #        "episodic_avg_val_loss": [], "episodic_avg_policy_loss": [], "epoch_score": [],
        #        "epoch_val_loss": [], "epoch_policy_loss": []}
        for epoch in epochs:
            for ep in train_episodes:
                done=False
                state=self.env.reset()
                while not done:
                    action, logproba = self.actor.get_action(state)
                    new_state, new_reward, done, info = self.env.step(action)
                    # delta = new_reward  + self.critic(new_state) - self.critic(state)
                    #In update policy, calculate loss and use adam optimizer
                    if not self.ppo:
                        self.critic.update_value(new_state, state, new_reward)
                        self.actor.update_policy(new_state, state, new_reward, logproba)
                    state=new_state

                    # self.log["value_loss"].append(self.critic_network.loss)
                    # self.log["policy_loss"].append(self.actor_network.loss)
    def test(self, test_episodes):
        # self.log = {"value_loss": [], "policy_loss": [], "score": [], "episodic_score": [],
        #        "episodic_avg_val_loss": [], "episodic_avg_policy_loss": [], "epoch_score": [],
        #        "epoch_val_loss": [], "epoch_policy_loss": []}
        episodic_reward=[]

        for ep in test_episodes:
            total_reward = 0
            done=False
            state=self.env.reset()
            while not done:
                action, logproba = self.actor.get_action(state)
                new_state, new_reward, done, info = self.env.step(action)
                state=new_state
                total_reward+=new_reward
            episodic_reward.append(total_reward)
        average_reward=np.mean(episodic_reward)

        return average_reward, episodic_reward











