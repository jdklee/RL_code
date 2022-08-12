from collections import deque
import random
import gym
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from utils import write_to_file, TradingGraph
from crypto_model_ppo import *

class custonEnv(gym.Env):
    def __init__(self, df, lookback_window=50,  initial_balance=1000, render_range=100):
        super().__init__()
        self.render_range=render_range
        self.env_name="Custom_Crypto_Trader"
        self.df = df.dropna().reset_index()
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.df_total_steps = len(df) - 1
        self.actions = [1, 2, 3]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.state_shape = (self.lookback_window, 10)
        self.orders_history = deque(maxlen=self.lookback_window)
        self.market_history = deque(maxlen=self.lookback_window)
        # self.Actor=Actor(input_shape=self.state_shape, action_space=self.actions, lr=0.00001, n_layers=10,
        #                  size=512, batch_size=64, logger=None, output_path="output")

    def reset(self, env_steps_size=0):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.episode_orders = 0
        self.visualization = TradingGraph(render_range=self.render_range)
        self.trades=deque(maxlen=self.render_range)


        if env_steps_size > 0:
            self.start_step = random.randint(self.lookback_window, self.df_total_steps - env_steps_size)
            self.end_step = self.df_total_steps
        else:
            self.start_step = self.lookback_window
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window)):
            current_step = self.current_step - i
            self.orders_history.append(
                [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            self.market_history.append([self.df.loc[current_step, 'Open'],
                                        self.df.loc[current_step, "High"],
                                        self.df.loc[current_step, "Low"],
                                        self.df.loc[current_step, "Close"],
                                        self.df.loc[current_step, "Volume"]])

        # Initial State returned
        state = np.concatenate((self.market_history, self.orders_history), axis=1).flatten()

        return state
    def create_writer(self):
        self.replay_count = 0
        self.writer = SummaryWriter(comment="Crypto_trader")




    def step(self, action):
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.current_step += 1
        try:
            current_price = random.uniform(self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])
            date = self.df.loc[self.current_step, "Date"]
            high = self.df.loc[self.current_step, "High"]
            low = self.df.loc[self.current_step,"Low"]

            if action == 0:
                pass
            if action == 1:  # buy with all balance
                self.crypto_bought = self.balance / current_price
                self.balance -= self.crypto_bought * current_price
                self.crypto_held += self.crypto_bought
                # self.net_worth += self.crypto_bought * current_price + self.balance
                self.episode_orders+=1
                self.trades.append({"Date":date, "High":high, "Low":low, "total":self.crypto_bought, "type":"buy"})

            elif action == 2:  # sell with all balance
                self.crypto_sold = self.crypto_held
                self.crypto_held -= self.crypto_sold
                self.balance += self.crypto_sold * current_price
                self.episode_orders += 1
                self.trades.append({"Date":date, "High":high, "Low":low, "total":self.crypto_bought, "type":"sell"})

            self.prev_net_worth = self.net_worth
            self.net_worth = self.balance + self.crypto_held * current_price
            order = [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held]
            self.orders_history.append(order)

            write_to_file(date, order)

            reward = self.net_worth - self.prev_net_worth

            if self.net_worth <= self.initial_balance / 2:
                done = True
            else:
                done = False

            state = self._next_observation()

            return state, reward, done, None

        except:
            print(self.current_step)
            print(self.lookback_window)
            print(self.df_total_steps)

    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'Open'],
                                    self.df.loc[self.current_step, "High"],
                                    self.df.loc[self.current_step, "Low"],
                                    self.df.loc[self.current_step, "Close"],
                                    self.df.loc[self.current_step, "Volume"]])

        state = np.concatenate((self.market_history, self.orders_history), axis=1).flatten()
        return state

    def render(self, visualize=False):
        if visualize:
            date = self.df.loc[self.current_step, "Date"]
            high = self.df.loc[self.current_step, "High"]
            low = self.df.loc[self.current_step, "Low"]
            open = self.df.loc[self.current_step, "Open"]
            close = self.df.loc[self.current_step, "Close"]

            volume = self.df.loc[self.current_step, "Volume"]

            self.visualization.render(date=date, high=high, low=low, open=open, close=close,
                                      volume=volume, net_worth = self.net_worth, trades=self.trades)
        # print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')






def Random_games(env, train_episodes=50, training_batch_size=500):
    average_net_worth = 0
    for episode in range(train_episodes):
        state = env.reset(env_steps_size=training_batch_size)

        while True:
            env.render()

            action = np.random.randint(3, size=1)[0]

            state, reward, done, _ = env.step(action)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    print("average_net_worth:", average_net_worth / train_episodes)
df = pd.read_csv('./eth_data.csv')
df = df.sort_values('Date')

lookback_window_size = 30
train_df = df[:-720-lookback_window_size]
test_df = df[-720-lookback_window_size:] # 30 days



train_env = custonEnv(train_df, lookback_window=lookback_window_size)
test_env = custonEnv(test_df, lookback_window=lookback_window_size)

Actor=Actor(env=train_env, input_shape=train_env.state_shape, gamma=0.9, action_space=train_env.actions, lr=0.00001,
            n_layers=10, size=512, batch_size=500, logger=None, output_path="output")
Actor.train(visualize=False, train_episodes=1500, training_batch_size=500)
Actor.test_agent()
# Random_games(train_env, train_episodes = 10, training_batch_size=500)