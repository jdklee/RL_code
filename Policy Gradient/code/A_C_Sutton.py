import torch
import numpy as np
import matplotlib.pyplot as plt


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
    def __init__(self, gamma, learning_rate, net_type, state_space, size, n_layers):
        """Initializes the critic network which approximates state value function,
        which is an average of value of all actions available in a given state"""
        super().__init__()
        self.lr = learning_rate
        self.net_type = net_type
        self.state_space, self.size, self.n_layers = state_space, size, n_layers
        self.action_space = 1
        self.gamma = gamma
        if net_type == 'linear':
            # Flatten state vector for linear layer
            state_space = state_space
            net = [torch.nn.Linear(state_space, size)]
            for n_layer in range(n_layers):
                net.append(torch.nn.Linear(size, size))
                net.append(torch.nn.ReLU())
            net.append(torch.nn.Linear(size, self.action_space))
            self.network = torch.nn.Sequential(*net)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, state):
        state = np2torch(state)
        value = self.network(state)
        return value

    def update_value(self, state, new_state, reward):

        # self.loss=torch.nn.CrossEntropyLoss(reward+self.gamma*self.critic(new_state), self.critic(state))
        self.loss = torch.nn.functional.mse_loss(reward + self.gamma * self.forward(new_state), self.forward(state))
        # self.loss = reward + self.gamma * self.forward(new_state) - self.forward(state)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def update_value_ppo(self,paths):
        states=np.concatenate([path["state"] for path in paths])
        advantage=np.concatenate([path["advantage"] for path in paths])
        advantage = np2torch(advantage)
        # self.loss=torch.nn.CrossEntropyLoss(reward+self.gamma*self.critic(new_state), self.critic(state))
        value_loss = torch.nn.functional.mse_loss(advantage, self.forward(states))
        # self.loss = reward + self.gamma * self.forward(new_state) - self.forward(state)
        self.loss=value_loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


class actor(torch.nn.Module):
    def __init__(self, gamma, actor_learning_rate, critic_learning_rate,
                 state_space, action_space, size, n_layers, ppo=False, net_type='linear',
                 discrete_action=False, discrete_state=False):
        super().__init__()
        self.lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.net_type = net_type
        self.ppo = ppo
        self.discrete_action = discrete_action
        self.discrete_state = discrete_state
        self.gamma = gamma
        self.ppo_eps=0.2
        #             self.discrete_action=discrete_action
        #             self.discrete_state=discrete_state
        #             self.action_dim = (
        #                 self.env.action_space.n if self.discrete_action else self.env.action_space.shape[0]
        #             )
        self.state_space, self.action_space, self.size, self.n_layers = state_space, action_space, size, n_layers
        if net_type == 'linear':
            # Flatten state vector for linear layer
            #                 state_space=np.multiply(state_space)
            net = [torch.nn.Linear(state_space, size)]
            for n_layer in range(n_layers):
                net.append(torch.nn.Linear(size, size))
                net.append(torch.nn.ReLU())
            net.append(torch.nn.Linear(size, self.action_space))
            self.network = torch.nn.Sequential(*net)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.critic = Critic(gamma=gamma, learning_rate=critic_learning_rate,
                             net_type=net_type, state_space=state_space,
                             size=size, n_layers=n_layers)

    def forward(self, state):
        # if type(state)==torch.tensor()
        state = np2torch(state)
        logits = self.network(state)
        return logits

    def get_action(self, state):
        logits = self.forward(state)

        if not self.discrete_action:
            distribution = torch.distributions.Categorical(logits=logits)
            sampled_action = distribution.sample()
            log_proba = distribution.log_prob(sampled_action).detach().numpy()
        else:  # Gaussian if continuous action
            log_std = torch.nn.Parameter(torch.zeros(size=[self.action_space]))
            std = np.exp(log_std.detach().numpy())
            std = np2torch(std)
            distribution = torch.distributions.MultivariateNormal(loc=logits,
                                                                  scale_tril=torch.diag(std))
            sampled_action = distribution.sample().detach().numpy()
            log_proba = distribution.log_prob(sampled_action).detach().numpy()

        return sampled_action, log_proba

    def update_policy_ppo(self, paths):
        states=np.concatenate([path["state"] for path in paths])
        action=np.concatenate([path["action"] for path in paths])
        advantage=np.concatenate([path["advantage"] for path in paths])
        reward = np.concatenate([path["reward"] for path in paths])
        batch_log_prob = np.concatenate([path["log_prob"] for path in paths])
        # states = np2torch(states)
        action = np2torch(action)
        advantage = np2torch(advantage)
        reward = np2torch(reward)
        batch_log_prob = np2torch(batch_log_prob)
        logits = self.forward(states)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probabilties = distribution.log_prob(action.reshape((-1,1)))
        advantage = advantage - self.critic.forward(states)
        ratio= batch_log_prob-log_probabilties
        ratio = torch.exp(ratio)

        loss1=ratio*advantage
        loss2=torch.clamp(ratio,1-self.ppo_eps, 1+self.ppo_eps) *advantage
        policy_loss = (-torch.min(loss1, loss2)).mean()
        self.loss=policy_loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()



    def update_policy(self, state, new_state, reward, logproba):
        if not self.ppo:
            # self.loss=torch.nn.CrossEntropyLoss(reward+self.gamma*self.critic(new_state), self.critic(state))
            self.loss = reward + self.gamma * self.critic(new_state) - self.critic(state)
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()


class Actor_critic_agent:
    # Takes in environment, critic network, actor network.
    def __init__(self, env, actor,
                 eligibility_trace=False, ppo=False):

        self.eligibility_trace = eligibility_trace
        self.env = env
        self.start_state = env.reset()
        #         self.critic=critic
        self.actor = actor
        self.ppo = ppo
        self.gamma=1

    def collect_path(self, batch_size):
        paths=[]
        episodic_score=[]
        for t in range(batch_size):
            done=False
            # done = False
            state = self.env.reset()
            score=0
            states, actions, advantages, rewards, log_probs = [],[],[],[], []
            while not done:
                states.append(state)
                action, log_prob = self.actor.get_action(state)
                action=action.numpy()
                actions.append(action)
                log_probs.append(log_prob)
                new_state, reward, done, info = self.env.step(action)
                rewards.append(reward)
                advantage=reward+self.gamma*self.actor.critic(new_state)
                advantages.append(advantage.detach().numpy())
                state=new_state
                score+=reward
            path={"state":states, "action":actions, "advantage":advantages, "reward":rewards, "log_prob":log_probs}
            paths.append(path)
            episodic_score.append(score)
        return paths, episodic_score


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

    def train(self, train_episodes):
        self.log = {"value_loss": [], "policy_loss": [], "score": [], "episodic_score": [],
                    "episodic_avg_val_loss": [], "episodic_avg_policy_loss": [], "epoch_score": [],
                    "epoch_val_loss": [], "epoch_policy_loss": []}
        for ep in range(train_episodes):
            # print(ep)
            self.env.render()
            done = False
            state = self.env.reset()
            score = 0
            val_loss = []
            pol_loss = []
            if not self.ppo:
                while not done:
                    env.render()
                    action, logproba = self.actor.get_action(state)
                    action = action.numpy()
                    new_state, new_reward, done, info = self.env.step(action)
                    # delta = new_reward  + self.critic(new_state) - self.critic(state)
                    # In update policy, calculate loss and use adam optimizer
                    # if not self.ppo:
                    self.actor.critic.update_value(new_state, state, new_reward)
                    self.actor.update_policy(new_state, state, new_reward, logproba)
                    state = new_state
                    score += new_reward
                    if done:
                        break
            else:
                paths, episodic_scores=self.collect_path(64)
                self.actor.update_policy_ppo(paths)
                self.actor.critic.update_value_ppo(paths)
                score=np.mean(episodic_scores)


            print("episode: {}, episodic score: {}".format(ep, score))
            self.log["episodic_score"].append(score)

    def test(self, test_episodes):
        self.test_log = {"episodic_score": []}
        episodic_reward = []

        for ep in range(test_episodes):
            total_reward = 0
            done = False
            state = self.env.reset()
            while not done:
                action, logproba = self.actor.get_action(state)
                action = action.numpy()
                new_state, new_reward, done, info = self.env.step(action)
                state = new_state
                total_reward += new_reward
            episodic_reward.append(total_reward)
        average_reward = np.mean(episodic_reward)

        return average_reward, episodic_reward
import torch
import numpy as np
import matplotlib.pyplot as plt



import gym

env = gym.make('CartPole-v1')
# print(env.action_space) #[Output: ] Discrete(2)
# print(env.observation_space) # [Output: ] Box(4,)

actor=actor( gamma=1, actor_learning_rate=1e-5, critic_learning_rate=1e-5,
                      state_space=env.observation_space.shape[0], size=128, n_layers=3, action_space=env.action_space.n, ppo=True)

A_C_Agent=Actor_critic_agent(env=env, actor=actor,
                 eligibility_trace=False, ppo=True)

A_C_Agent.train(2000)
# plt.plot(range(len(A_C_Agent.log['episodic_score'])), A_C_Agent.log['episodic_score'])
# plt.show()
# plt.savefig("episodic_score_actor_acritic")
# plt.close()

average_reward, episodic_reward=A_C_Agent.test(10)
print(average_reward)
plt.plot(range(len(episodic_reward)), episodic_reward)
plt.show()
plt.savefig("test_episodic_score_actor_critic.png")
plt.close()