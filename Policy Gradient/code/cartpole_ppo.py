import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import gym

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
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, size, n_layers, dropout=0.5, lr=0.01):
        super().__init__()
        net=[nn.Linear(input_dim, size),nn.Dropout(dropout),nn.ReLU()]
        for _ in range(n_layers):
            net.append(nn.Linear(size,size))
            net.append(nn.Dropout(dropout))
            net.append(nn.ReLU())
        net.append(nn.Linear(size, output_dim))


        self.net=nn.Sequential(*net)

    def forward(self, x):
        try:
            np2torch(x)
        except:
            pass
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()

        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)

        return action_pred, value_pred
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
def get_returns(rewards, gamma, normalize=True):
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

    TODO: compute and return G_t for each timestep. Use self.config.gamma.
    """

    all_returns = []
    R=0
    for r in reversed(rewards):
        R=r+R*gamma
        all_returns.insert(0, R)
    # all_returns.append(returns)
    returns = torch.tensor(all_returns)
    if normalize:
        returns= (returns - returns.mean())/returns.std()
    return returns


def train(env, agent, optimizer, gamma, ppo_steps, ppo_clip=0.2):
    agent.train()
    states, actions, log_probs, values, rewards = [],[],[],[],[]
    done=False
    ep_reward=0
    state=env.reset()
    env.render()
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        states.append(state)
        action_prediction, value_prediction = agent(state)
        action_prob = F.softmax(action_prediction, dim=-1)
        dist=distributions.Categorical(action_prob)
        action=dist.sample()
        log_prob=dist.log_prob(action)
        state,reward,done,_=env.step(action.item())
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value_prediction)
        rewards.append(reward)
        ep_reward+=reward
    states=torch.cat(states)
    actions=torch.cat(actions)
    log_probs=torch.cat(log_probs)
    values = torch.cat(values).squeeze(-1)

    returns=get_returns(rewards, gamma=0.9)

    advantages=returns-values
    advantages=(advantages-advantages.mean())/advantages.std()

    policy_loss, value_loss = update_policy(agent, states, actions, log_probs, advantages,
                                            returns, optimizer, ppo_steps, ppo_clip)

    return policy_loss, value_loss, ep_reward

def update_policy(agent, states, actions, log_probs, advantages,
                                            returns, optimizer, ppo_steps, ppo_clip):

    total_policy_loss=0
    total_value_loss=0
    advantages = advantages.detach()
    log_probs = log_probs.detach()
    actions = actions.detach()


    for _ in range(ppo_steps):
        action_pred, value_pred = agent(states)
        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim=-1)
        dist=distributions.Categorical(action_prob)
        #Get log prob using old actions for old log probabilities
        new_log_prob_from_old_actions=dist.log_prob(actions)

        ratio = (new_log_prob_from_old_actions - log_probs).exp()
        surrogate_loss_1 = ratio * advantages
        surrogate_loss_2 = torch.clamp(ratio, 1 - ppo_clip, 1 +ppo_clip) * advantages
        policy_loss = (-torch.min(surrogate_loss_2, surrogate_loss_1)).sum()
        value_loss = torch.nn.functional.mse_loss(returns, value_pred).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        optimizer.step()

        total_policy_loss+=policy_loss.item()
        total_value_loss+=value_loss.item()
    return total_policy_loss/ppo_steps, total_value_loss/ppo_steps



def evaluate(env, agent):
    agent.eval()
    rewards=[]
    done=False
    ep_rew=0
    state=env.reset()
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_pred, value_prd = agent(state)
            action_prob=F.softmax(action_pred, dim=-1)
        action=torch.argmax(action_prob, dim=-1)
        state,reward,done,_ = env.step(action.item())
        ep_rew+=reward

    return ep_rew

#######################################################################################################################
#################################Variables#############################################################################
#######################################################################################################################
train_env = gym.make('CartPole-v1')
test_env = gym.make('CartPole-v1')
SEED = 1234
train_env.seed(SEED)
test_env.seed(SEED+1)
np.random.seed(SEED)
torch.manual_seed(SEED)
INPUT_DIM = train_env.observation_space.shape[0]
# HIDDEN_DIM = 1
OUTPUT_DIM = train_env.action_space.n
SIZE=128
N_LAYERS=2
actor = MLP(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, size=SIZE, n_layers=N_LAYERS, dropout=0.5, lr=0.01)
critic = MLP(input_dim=INPUT_DIM, output_dim=1, size=SIZE, n_layers=N_LAYERS, dropout=0.5, lr=0.01)
agent = ActorCritic(actor, critic)
agent.apply(init_weights)
LEARNING_RATE=0.01
optimizer=optim.Adam(agent.parameters(), lr=LEARNING_RATE)

MAX_EPISODES = 300
DISCOUNT_FACTOR = 0.99
N_TRIALS = 25
REWARD_THRESHOLD = 475
PRINT_EVERY = 10
PPO_STEPS = 5
PPO_CLIP = 0.2

train_rewards = []
test_rewards = []

for episode in range(1, MAX_EPISODES + 1):

    policy_loss, value_loss, train_reward = train(env=train_env, agent=agent, optimizer=optimizer, gamma=DISCOUNT_FACTOR, ppo_steps=PPO_STEPS, ppo_clip=0.2)

    test_reward = evaluate(env=test_env, agent=agent)

    train_rewards.append(train_reward)
    test_rewards.append(test_reward)

    mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
    mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

    if episode % PRINT_EVERY == 0:
        print(
            f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')

    if mean_test_rewards >= REWARD_THRESHOLD:
        print(f'Reached reward threshold in {episode} episodes')

        break


plt.figure(figsize=(12,8))
plt.plot(test_rewards, label='Test Reward')
plt.plot(train_rewards, label='Train Reward')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r')
plt.legend(loc='lower right')
plt.grid()
plt.show()