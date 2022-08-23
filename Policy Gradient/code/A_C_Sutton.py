import numpy as np
import torch
import pytorch_lightning as pl
# from pytorch_lightning import LightningModule, Trainer
# from pytorch_lightning.callbacks.progress import TQDMProgressBar
# from pytorch_lightning.loggers import CSVLogger
import numpy as np
import matplotlib.pyplot as plt
import gym



def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x


class Critic(torch.nn.Module):
    def __init__(self, gamma, learning_rate, net_type, state_space, size, n_layers, drop_out=0.5):
        """Initializes the critic network which approximates state value function,
        which is an average of value of all actions available in a given state"""
        super().__init__()
        self.lr = learning_rate
        self.net_type = net_type
        self.state_space, self.size, self.n_layers = state_space, size, n_layers
        self.action_space = 1
        self.gamma = gamma
        self.drop_out=drop_out
        if net_type == 'linear':
            # Flatten state vector for linear layer
            state_space = state_space
            net = [torch.nn.Linear(state_space, size)]
            for n_layer in range(n_layers):
                net.append(torch.nn.Linear(size, size))
                net.append(torch.nn.Dropout(self.drop_out))
                net.append(torch.nn.ReLU())

            net.append(torch.nn.Linear(size, self.action_space))

        self.network = torch.nn.Sequential(*net)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, state):
        # state = np2torch(state)
        value = self.network(state)
        return value

    # def update_value(self, loss):
    #
    #     # self.loss=torch.nn.CrossEntropyLoss(reward+self.gamma*self.critic(new_state), self.critic(state))
    #     # self.loss = torch.nn.functional.mse_loss(reward + self.gamma * self.forward(new_state), self.forward(state))
    #     # self.loss = reward + self.gamma * self.forward(new_state) - self.forward(state)
    #     self.loss=loss
    #     self.optimizer.zero_grad()
    #     self.loss.backward()
    #     self.optimizer.step()

    # def update_value(self, state, new_state, reward):
    #
    #     # self.loss=torch.nn.CrossEntropyLoss(reward+self.gamma*self.critic(new_state), self.critic(state))
    #     self.loss = torch.nn.functional.mse_loss(reward + self.gamma * self.forward(new_state), self.forward(state))
    #     # self.loss = reward + self.gamma * self.forward(new_state) - self.forward(state)
    #     self.optimizer.zero_grad()
    #     self.loss.backward()
    #     self.optimizer.step()
    #
    # def update_value_ppo(self,paths):
    #     states=np.concatenate([path["state"] for path in paths])
    #     advantage=np.concatenate([path["advantage"] for path in paths])
    #     advantage = np2torch(advantage)
    #     # self.loss=torch.nn.CrossEntropyLoss(reward+self.gamma*self.critic(new_state), self.critic(state))
    #     value_loss = torch.nn.functional.mse_loss(advantage, self.forward(states))
    #     # self.loss = reward + self.gamma * self.forward(new_state) - self.forward(state)
    #     self.loss=value_loss
    #     self.optimizer.zero_grad()
    #     self.loss.backward()
    #     self.optimizer.step()


class actor(torch.nn.Module):
    def __init__(self, gamma, actor_learning_rate, critic_learning_rate,
                 state_space, action_space, size, n_layers, ppo=False, net_type='linear',
                 discrete_action=False, discrete_state=False, drop_out=0.5):
        super().__init__()
        self.lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.net_type = net_type
        self.ppo = ppo
        self.discrete_action = discrete_action
        self.discrete_state = discrete_state
        self.gamma = gamma
        self.ppo_eps = 0.2
        self.drop_out=drop_out
        #             self.discrete_action=discrete_action
        #             self.discrete_state=discrete_state
        #             self.action_dim = (
        #                 self.env.action_space.n if self.discrete_action else self.env.action_space.shape[0]
        #             )
        self.state_space, self.action_space, self.size, self.n_layers = state_space, action_space, size, n_layers
        if net_type == 'linear':
            net = [torch.nn.Linear(state_space, size)]
            for n_layer in range(n_layers):
                net.append(torch.nn.Linear(size, size))
                net.append(torch.nn.Dropout(self.drop_out))
                net.append(torch.nn.ReLU())

            net.append(torch.nn.Linear(size, self.action_space))
            self.network = torch.nn.Sequential(*net)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.critic = Critic(gamma=gamma, learning_rate=critic_learning_rate,
                             net_type=net_type, state_space=state_space,
                             size=size, n_layers=n_layers, drop_out=self.drop_out)

    def forward(self, state):
        # if type(state)==torch.tensor()
        # state = np2torch(state)
        logits = self.network(state)
        return logits

    # def get_action(self, state):
    #     logits = self.forward(state)
    #     logits=torch.nn.functional.softmax(logits, dim=-1)
    #
    #
    #     if not self.discrete_action:
    #         distribution = torch.distributions.Categorical(logits=logits)
    #         sampled_action = distribution.sample()
    #         log_proba = distribution.log_prob(sampled_action).detach().numpy()
    #     else:  # Gaussian if continuous action
    #         log_std = torch.nn.Parameter(torch.zeros(size=[self.action_space]))
    #         std = np.exp(log_std.detach().numpy())
    #         std = np2torch(std)
    #         distribution = torch.distributions.MultivariateNormal(loc=logits,
    #                                                               scale_tril=torch.diag(std))
    #         sampled_action = distribution.sample().detach().numpy()
    #         log_proba = distribution.log_prob(sampled_action).detach().numpy()
    #
    #     return sampled_action, log_proba
    #
    # def update_policy_ppo(self, paths):
    #     states=np.concatenate([path["state"] for path in paths])
    #     action=np.concatenate([path["action"] for path in paths])
    #     advantage=np.concatenate([path["advantage"] for path in paths])
    #     reward = np.concatenate([path["reward"] for path in paths])
    #     batch_log_prob = np.concatenate([path["log_prob"] for path in paths])
    #     # states = np2torch(states)
    #     action = np2torch(action)
    #     advantage = np2torch(advantage)
    #     reward = np2torch(reward)
    #     batch_log_prob = np2torch(batch_log_prob)
    #     logits = self.forward(states)
    #     distribution = torch.distributions.Categorical(logits=logits)
    #     log_probabilties = distribution.log_prob(action.reshape((-1,1)))
    #     advantage = advantage - self.critic.forward(states)
    #     ratio= batch_log_prob-log_probabilties
    #     ratio = torch.exp(ratio)
    #
    #     loss1=ratio*advantage
    #     loss2=torch.clamp(ratio,1-self.ppo_eps, 1+self.ppo_eps) *advantage
    #     policy_loss = (-torch.min(loss1, loss2)).mean()
    #     self.loss=policy_loss
    #     self.optimizer.zero_grad()
    #     self.loss.backward()
    #     self.optimizer.step()

    # def update_policy(self, loss):
    #     if not self.ppo:
    #         # self.loss=torch.nn.CrossEntropyLoss(reward+self.gamma*self.critic(new_state), self.critic(state))
    #         # self.loss = reward + self.gamma * self.critic(new_state) - self.critic(state)
    #         self.loss=loss
    #         self.optimizer.zero_grad()
    #         self.loss.backward()
    #         self.optimizer.step()


class Actor_critic_agent(torch.nn.Module):
    # Takes in environment, critic network, actor network.
    def __init__(self, actor, critic, lr, episodic=True, max_ep_len=5000):
        super().__init__()
        self.lr = lr
        self.actor = actor
        self.critic = critic
        self.episodic=episodic
        self.gamma = 1
        self.max_ep_len = max_ep_len
        self.ppo_steps = 5
        self.ppo_clip = 0.2
        self.discrete_actions = False
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred

    def calculate_returns(self, rewards, normalize=True):
        all_returns = []
        R = 0
        for r in reversed(rewards):
            R = r + R * self.gamma
            all_returns.insert(0, R)
        # all_returns.append(returns)
        returns = torch.tensor(all_returns)
        if normalize:
            returns = (returns - returns.mean()) / returns.std()
        return returns

    def calculate_advantage(self, returns, values):
        a = returns - values
        a = (a - a.mean()) / a.std()
        return a

    def collect_path(self):

        # for t in range(batch_size):
        done = False
        # done = False
        state = self.env.reset()
        score = 0
        states, actions, rewards, log_probs, values = [], [], [], [], []
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            states.append(state)
            action_pred, value_pred = self.forward(state)
            logits = torch.nn.functional.softmax(action_pred, dim=-1)
            if not self.discrete_actions:
                dist = torch.distributions.Categorical(logits)

            else:
                log_std = torch.nn.Parameter(torch.zeros(size=[self.action_space]))
                std = np.exp(log_std.detach().numpy())
                std = np2torch(std)
                dist = torch.distributions.MultivariateNormal(loc=logits,
                                                              scale_tril=torch.diag(std))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.actor.critic(state)
            actions.append(action)
            log_probs.append(log_prob)
            state, reward, done, info = self.env.step(action.item())
            rewards.append(reward)
            values.append(value)
            score += reward

        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        values = torch.cat(values).squeeze(-1)
        returns = self.calculate_returns(rewards)
        advantages = self.calculate_advantage(returns, values)

        return states, actions, log_probs, values, returns, advantages, score

    def export_plot(self, ys, ylabel, filename):
        """
        Export a plot in filename

        Args:
            ys: (list) of float / int to plot
            filename: (string) directory
        """
        plt.figure()
        plt.plot(range(len(ys)), ys)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.savefig(filename)
        plt.close()

    def train_agent(self, train_env):
        self.train()
        done = False
        # done = False
        state = train_env.reset()
        score = 0
        states, actions, rewards, log_probs, values = [], [], [], [], []
        num_events_seen = 0
        while not done:
            if not self.episodic and num_events_seen >= self.max_ep_len:
                break
            state = torch.FloatTensor(state).unsqueeze(0)
            states.append(state)
            action_pred, value_pred = self.forward(state)
            logits = torch.nn.functional.softmax(action_pred, dim=-1)
            if not self.discrete_actions:
                dist = torch.distributions.Categorical(logits)

            else:
                log_std = torch.nn.Parameter(torch.zeros(size=[self.action_space]))
                std = np.exp(log_std.detach().numpy())
                std = np2torch(std)
                dist = torch.distributions.MultivariateNormal(loc=logits,
                                                              scale_tril=torch.diag(std))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.actor.critic(state)
            actions.append(action)
            log_probs.append(log_prob)
            state, reward, done, info = train_env.step(action.item())
            rewards.append(reward)
            values.append(value)
            score += reward
            num_events_seen += 1

        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        values = torch.cat(values).squeeze(-1)
        returns = self.calculate_returns(rewards)
        advantages = self.calculate_advantage(returns, values)
        policy_loss, value_loss = self.update_network(states, actions, log_probs, advantages, returns)

        return policy_loss, value_loss, score

    def update_network(self, states, actions, log_probs, advantages, returns):
        advantages = advantages.detach()
        log_probs = log_probs.detach()
        actions = actions.detach()
        total_policy_loss = 0
        total_value_loss = 0
        for _ in range(self.ppo_steps):
            # Calculate policy loss
            logits = self.actor(states)
            logits = torch.nn.functional.softmax(logits, dim=-1)
            if not self.discrete_actions:
                dist = torch.distributions.Categorical(logits)

            else:
                log_std = torch.nn.Parameter(torch.zeros(size=[self.action_space]))
                std = np.exp(log_std.detach().numpy())
                std = np2torch(std)
                dist = torch.distributions.MultivariateNormal(loc=logits,
                                                              scale_tril=torch.diag(std))
            new_log_prob_from_old_actions = dist.log_prob(actions)
            ratio = (new_log_prob_from_old_actions - log_probs).exp()
            loss1 = ratio * advantages
            loss2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
            policy_loss = (-torch.min(loss2, loss1)).sum()

            # calculate value loss
            value_pred = self.actor.critic(states)
            returns = returns.reshape([-1, 1])
            value_pred = value_pred.reshape([-1, 1])
            value_loss = torch.nn.functional.mse_loss(returns, value_pred).sum()

            # update both value and policy
            # self.actor.update_policy(policy_loss)
            # self.actor.critic.update_value(value_loss)

            self.optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.optimizer.step()

            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()

        return total_policy_loss / self.ppo_steps, total_value_loss / self.ppo_steps

    def test_agent(self, test_env):
        self.eval()

        # self.test_log = {"episodic_score": []}
        episodic_reward = 0

        done = False
        state = test_env.reset()
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_pred = self.actor(state)
                action_prob = torch.nn.functional.softmax(action_pred, dim=-1)
            action = torch.argmax(action_prob, dim=-1)
            state, reward, done, info = test_env.step(action.item())
            episodic_reward += reward
        return episodic_reward


import torch


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()

train_env = gym.make('CartPole-v1')
test_env = gym.make('CartPole-v1')
SEED = 1234
train_env.seed(SEED)
test_env.seed(SEED)

MAX_EPISODES = 5000
DISCOUNT_FACTOR = 0.99
N_TRIALS = 25
REWARD_THRESHOLD = 400
PRINT_EVERY = 10
PPO_STEPS = 5
PPO_CLIP = 0.2
LEARNING_RATE = 0.01
DROP_OUT_RATE=0.3
SIZE_OF_HIDDEN_NETWORK=128
NUMBER_OF_LAYERS=2
# actor = actor(gamma=DISCOUNT_FACTOR, actor_learning_rate=LEARNING_RATE, critic_learning_rate=LEARNING_RATE,
#               state_space=train_env.observation_space.shape[0], size=SIZE_OF_HIDDEN_NETWORK, n_layers=NUMBER_OF_LAYERS,
#               action_space=train_env.action_space.n, ppo=True, drop_out=DROP_OUT_RATE)
# critic = actor.critic
# A_C_Agent = Actor_critic_agent(actor=actor, critic=critic, lr=LEARNING_RATE, episodic=True)
# A_C_Agent.apply(init_weights)

done=False
while not done:
    train_rewards = []
    test_rewards = []
    prev_mean_train_rewards = 0
    prev_mean_test_rewards = 0
    grace_count = 0
    ACTOR = actor(gamma=DISCOUNT_FACTOR, actor_learning_rate=LEARNING_RATE, critic_learning_rate=LEARNING_RATE,
                  state_space=train_env.observation_space.shape[0], size=SIZE_OF_HIDDEN_NETWORK,
                  n_layers=NUMBER_OF_LAYERS,
                  action_space=train_env.action_space.n, ppo=True, drop_out=DROP_OUT_RATE)
    CRITIC = ACTOR.critic
    A_C_Agent = Actor_critic_agent(actor=ACTOR, critic=CRITIC, lr=LEARNING_RATE, episodic=True)
    A_C_Agent.apply(init_weights)
    for episode in range(1, MAX_EPISODES + 1):
        policy_loss, value_loss, train_reward = A_C_Agent.train_agent(train_env=train_env)
        test_reward = A_C_Agent.test_agent(test_env=test_env)

        train_rewards.append(train_reward)
        test_rewards.append(test_reward)


        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        if prev_mean_test_rewards< mean_test_rewards:
            grace_count+=1
            if grace_count>50:
                grace_count=0
                # A_C_Agent.apply(weight_reset)
                print("reset network")
                break

        prev_mean_train_rewards = mean_train_rewards
        prev_mean_test_rewards = mean_test_rewards

        if episode % PRINT_EVERY == 0:
            print(
                f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')

        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            done=True

            break

plt.figure(figsize=(12, 8))
plt.plot(test_rewards, label='Test Reward')
plt.plot(train_rewards, label='Train Reward')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r')
plt.legend(loc='lower right')
plt.savefig("A_C_Sutton_results.png")
plt.grid()
# plt.show()
