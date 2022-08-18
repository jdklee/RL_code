# First we implement a basic DQN agent using fully connected linear neural networks
import copy
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from DQN.utils.test_env import EnvTest
from DQN.core.deep_q_learning_torch import DQN
from DQN.q4_schedule import LinearExploration, LinearSchedule

from configs.q5_linear import config


class Linear(DQN):
    def __init__(self, env, config, logger=None):
        if not os.path.exists(config.output_path):

        os.makedirs(config.output_path)

        # store hyper params
        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env
        self.timer = Timer(False)
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f'Running model on device {self.device}')
        super().__init__(env, config, logger)
        self.summary_writer = SummaryWriter(self.config.output_path, max_queue=1e5)

        # build model
        self.build()





def get_action(self, state):
    """
    Returns action with some epsilon strategy

    Args:
        state: observation from gym
    """
    if np.random.random() < self.config.soft_epsilon:
        return self.env.action_space.sample()
    else:
        return self.get_best_action(state)[0]



def init_averages(self):
    """
    Defines extra attributes for tensorboard
    """
    self.avg_reward = -21.
    self.max_reward = -21.
    self.std_reward = 0

    self.avg_q = 0
    self.max_q = 0
    self.std_q = 0

    self.eval_reward = -21.


def update_averages(self, rewards, max_q_values, q_values, scores_eval):
    """
    Update the averages

    Args:
        rewards: deque
        max_q_values: deque
        q_values: deque
        scores_eval: list
    """
    self.avg_reward = np.mean(rewards)
    self.max_reward = np.max(rewards)
    self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

    self.max_q = np.mean(max_q_values)
    self.avg_q = np.mean(q_values)
    self.std_q = np.sqrt(np.var(q_values) / len(q_values))

    if len(scores_eval) > 0:
        self.eval_reward = scores_eval[-1]


def add_summary(self, latest_loss, latest_total_norm, t):
    pass


def train(self, exp_schedule, lr_schedule):
    """
    Performs training of Q

    Args:
        exp_schedule: Exploration instance s.t.
            exp_schedule.get_action(best_action) returns an action
        lr_schedule: Schedule for learning rate
    """

    # initialize replay buffer and variables
    replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
    rewards = deque(maxlen=self.config.num_episodes_test)
    max_q_values = deque(maxlen=1000)
    q_values = deque(maxlen=1000)
    self.init_averages()

    t = last_eval = last_record = 0  # time control of nb of steps
    scores_eval = []  # list of scores computed at iteration time
    scores_eval += [self.evaluate()]

    prog = Progbar(target=self.config.nsteps_train)

    # interact with environment
    while t < self.config.nsteps_train:
        total_reward = 0
        self.timer.start('env.reset')
        state = self.env.reset()
        self.timer.end('env.reset')
        while True:
            t += 1
            last_eval += 1
            last_record += 1
            if self.config.render_train: self.env.render()
            # replay memory stuff
            self.timer.start('replay_buffer.store_encode')
            idx = replay_buffer.store_frame(state)
            q_input = replay_buffer.encode_recent_observation()
            self.timer.end('replay_buffer.store_encode')

            # chose action according to current Q and exploration
            self.timer.start('get_action')
            best_action, q_vals = self.get_best_action(q_input)
            action = exp_schedule.get_action(best_action)
            self.timer.end('get_action')

            # store q values
            max_q_values.append(max(q_vals))
            q_values += list(q_vals)

            # perform action in env
            self.timer.start('env.step')
            new_state, reward, done, info = self.env.step(action)
            self.timer.end('env.step')

            # store the transition
            self.timer.start('replay_buffer.store_effect')
            replay_buffer.store_effect(idx, action, reward, done)
            state = new_state
            self.timer.end('replay_buffer.store_effect')

            # perform a training step
            self.timer.start('train_step')
            loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)
            self.timer.end('train_step')

            # logging stuff
            if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and
                    (t % self.config.learning_freq == 0)):
                self.timer.start('logging')
                self.update_averages(rewards, max_q_values, q_values, scores_eval)
                self.add_summary(loss_eval, grad_eval, t)
                exp_schedule.update(t)
                lr_schedule.update(t)
                if len(rewards) > 0:
                    prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg_R", self.avg_reward),
                                              ("Max_R", np.max(rewards)), ("eps", exp_schedule.epsilon),
                                              ("Grads", grad_eval), ("Max_Q", self.max_q),
                                              ("lr", lr_schedule.epsilon)], base=self.config.learning_start)
                self.timer.end('logging')
            elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
                sys.stdout.write("\rPopulating the memory {}/{}...".format(t,
                                                                           self.config.learning_start))
                sys.stdout.flush()
                prog.reset_start()

            # count reward
            total_reward += reward
            if done or t >= self.config.nsteps_train:
                break

        # updates to perform at the end of an episode
        rewards.append(total_reward)

        if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
            # evaluate our policy
            last_eval = 0
            print("")
            self.timer.start('eval')
            scores_eval += [self.evaluate()]
            self.timer.end('eval')
            self.timer.print_stat()
            self.timer.reset_stat()

        if (t > self.config.learning_start) and self.config.record and (last_record > self.config.record_freq):
            self.logger.info("Recording...")
            last_record = 0
            self.timer.start('recording')
            self.record()
            self.timer.end('recording')

    # last words
    self.logger.info("- Training done.")
    self.save()
    scores_eval += [self.evaluate()]
    export_plot(scores_eval, "Scores", self.config.plot_output)


def train_step(self, t, replay_buffer, lr):
    """
    Perform training step

    Args:
        t: (int) nths step
        replay_buffer: buffer for sampling
        lr: (float) learning rate
    """
    loss_eval, grad_eval = 0, 0

    # perform training step
    if (t > self.config.learning_start and t % self.config.learning_freq == 0):
        self.timer.start('train_step/update_step')
        loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)
        self.timer.end('train_step/update_step')

    # occasionaly update target network with q network
    if t % self.config.target_update_freq == 0:
        self.timer.start('train_step/update_param')
        self.update_target_params()
        self.timer.end('train_step/update_param')

    # occasionaly save the weights
    if (t % self.config.saving_freq == 0):
        self.timer.start('train_step/save')
        self.save()
        self.timer.end('train_step/save')

    return loss_eval, grad_eval


def evaluate(self, env=None, num_episodes=None):
    """
    Evaluation with same procedure as the training
    """
    # log our activity only if default call
    if num_episodes is None:
        self.logger.info("Evaluating...")

    # arguments defaults
    if num_episodes is None:
        num_episodes = self.config.num_episodes_test

    if env is None:
        env = self.env

    # replay memory to play
    replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
    rewards = []

    for i in range(num_episodes):
        total_reward = 0
        state = env.reset()
        while True:
            if self.config.render_test: env.render()

            # store last state in buffer
            idx = replay_buffer.store_frame(state)
            q_input = replay_buffer.encode_recent_observation()

            action = self.get_action(q_input)

            # perform action in env
            new_state, reward, done, info = env.step(action)

            # store in replay memory
            replay_buffer.store_effect(idx, action, reward, done)
            state = new_state

            # count reward
            total_reward += reward
            if done:
                break

        # updates to perform at the end of an episode
        rewards.append(total_reward)

    avg_reward = np.mean(rewards)
    sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

    if num_episodes > 1:
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)

    return avg_reward


def record(self):
    """
    Re create an env and record a video for one episode
    """
    env = gym.make(self.config.env_name)
    env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
    env = MaxAndSkipEnv(env, skip=self.config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                        overwrite_render=self.config.overwrite_render)
    self.evaluate(env, 1)


def run(self, exp_schedule, lr_schedule):
    """
    Apply procedures of training for a QN

    Args:
        exp_schedule: exploration strategy for epsilon
        lr_schedule: schedule for learning rate
    """
    # initialize
    self.initialize()

    # record one game at the beginning
    if self.config.record:
        self.record()

    # model
    self.train(exp_schedule, lr_schedule)

    # record one game at the end
    if self.config.record:
        self.record()


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

    self.q_network = torch.nn.Linear(img_height * img_width * n_channels * self.config.state_history, num_actions)
    self.target_network = torch.nn.Linear(img_height * img_width * n_channels * self.config.state_history, num_actions)


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

    if network == "q_network":
        out = self.q_network(torch.flatten(state, start_dim=1))
    else:
        out = self.target_network(torch.flatten(state, start_dim=1))

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
    self.target_network = torch.load("q_network_weights")


def calc_loss(self, q_values: Tensor, target_q_values: Tensor,
              actions: Tensor, rewards: Tensor, done_mask: Tensor) -> Tensor:
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
    num_actions = self.env.action_space.n
    actions_taken = torch.nn.functional.one_hot(actions.to(torch.int64), num_classes=num_actions)
    q_val_after_done = ~done_mask[:, None] * target_q_values

    max_q_target, max_q_target_idx = torch.max(q_val_after_done, dim=1)
    gamma = self.config.gamma
    q_samp = rewards + gamma * max_q_target
    q_sa = torch.sum(q_values * actions_taken, dim=1)

    loss = torch.nn.functional.mse_loss(q_samp, q_sa)
    return loss


def add_optimizer(self, lr=0.00001):
    """
    Set self.optimizer to be an Adam optimizer optimizing only the self.q_network
    parameters
    """
    self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)


def process_state(self, state: Tensor) -> Tensor:
    """
    Processing of state

    State placeholders are tf.uint8 for fast transfer to GPU
    Need to cast it to float32 for the rest of the tf graph.

    Args:
        state: node of tf graph of shape = (batch_size, height, width, nchannels)
                of type tf.uint8.
                if , values are between 0 and 255 -> 0 and 1
    """
    state = state.float()
    state /= self.config.high

    return state


def build(self):
    """
    Build model by adding all necessary variables
    """
    self.initialize_models()
    if hasattr(self.config, 'load_path'):
        print('Loading parameters from file:', self.config.load_path)
        load_path = Path(self.config.load_path)
        assert load_path.is_file(), f'Provided load_path ({load_path}) does not exist'
        self.q_network.load_state_dict(torch.load(load_path, map_location='cpu'))
        print('Load successful!')
    else:
        print('Initializing parameters randomly')

        def init_weights(m):
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight, gain=2 ** (1. / 2))
            if hasattr(m, 'bias'):
                nn.init.zeros_(m.bias)

        self.q_network.apply(init_weights)
    self.q_network = self.q_network.to(self.device)
    self.target_network = self.target_network.to(self.device)
    self.add_optimizer()


def initialize(self):
    """
    Assumes the graph has been constructed
    Creates a tf Session and run initializer of variables
    """
    # synchronise q and target_q networks
    assert self.q_network is not None and self.target_network is not None, \
        'WARNING: Networks not initialized. Check initialize_models'
    self.update_target()


def add_summary(self, latest_loss, latest_total_norm, t):
    """
    Tensorboard stuff
    """
    self.summary_writer.add_scalar('loss', latest_loss, t)
    self.summary_writer.add_scalar('grad_norm', latest_total_norm, t)
    self.summary_writer.add_scalar('Avg_Reward', self.avg_reward, t)
    self.summary_writer.add_scalar('Max_Reward', self.max_reward, t)
    self.summary_writer.add_scalar('Std_Reward', self.std_reward, t)
    self.summary_writer.add_scalar('Avg_Q', self.avg_q, t)
    self.summary_writer.add_scalar('Max_Q', self.max_q, t)
    self.summary_writer.add_scalar('Std_Q', self.std_q, t)
    self.summary_writer.add_scalar('Eval_Reward', self.eval_reward, t)


def save(self):
    """
    Saves session
    """
    # if not os.path.exists(self.config.model_output):
    #     os.makedirs(self.config.model_output)
    torch.save(self.q_network.state_dict(), self.config.model_output)
    # self.saver.save(self.sess, self.config.model_output)


def get_best_action(self, state: Tensor) -> Tuple[int, np.ndarray]:
    """
    Return best action

    Args:
        state: 4 consecutive observations from gym
    Returns:
        action: (int)
        action_values: (np array) q values for all actions
    """
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0)
        s = self.process_state(s)
        action_values = self.get_q_values(s, 'q_network').squeeze().to('cpu').tolist()
    action = np.argmax(action_values)
    return action, action_values


def update_step(self, t, replay_buffer, lr):
    """
    Performs an update of parameters by sampling from replay_buffer

    Args:
        t: number of iteration (episode and move)
        replay_buffer: ReplayBuffer instance .sample() gives batches
        lr: (float) learning rate
    Returns:
        loss: (Q - Q_target)^2
    """
    self.timer.start('update_step/replay_buffer.sample')
    s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
        self.config.batch_size)
    self.timer.end('update_step/replay_buffer.sample')

    assert self.q_network is not None and self.target_network is not None, \
        'WARNING: Networks not initialized. Check initialize_models'
    assert self.optimizer is not None, \
        'WARNING: Optimizer not initialized. Check add_optimizer'

    # Convert to Tensor and move to correct device
    self.timer.start('update_step/converting_tensors')
    s_batch = torch.tensor(s_batch, dtype=torch.uint8, device=self.device)
    a_batch = torch.tensor(a_batch, dtype=torch.uint8, device=self.device)
    r_batch = torch.tensor(r_batch, dtype=torch.float, device=self.device)
    sp_batch = torch.tensor(sp_batch, dtype=torch.uint8, device=self.device)
    done_mask_batch = torch.tensor(done_mask_batch, dtype=torch.bool, device=self.device)
    self.timer.end('update_step/converting_tensors')

    # Reset Optimizer
    self.timer.start('update_step/zero_grad')
    self.optimizer.zero_grad()
    self.timer.end('update_step/zero_grad')

    # Run a forward pass
    self.timer.start('update_step/forward_pass_q')
    s = self.process_state(s_batch)
    q_values = self.get_q_values(s, 'q_network')
    self.timer.end('update_step/forward_pass_q')

    self.timer.start('update_step/forward_pass_target')
    with torch.no_grad():
        sp = self.process_state(sp_batch)
        target_q_values = self.get_q_values(sp, 'target_network')
    self.timer.end('update_step/forward_pass_target')

    self.timer.start('update_step/loss_calc')
    loss = self.calc_loss(q_values, target_q_values,
                          a_batch, r_batch, done_mask_batch)
    self.timer.end('update_step/loss_calc')
    self.timer.start('update_step/loss_backward')
    loss.backward()
    self.timer.end('update_step/loss_backward')

    # Clip norm
    self.timer.start('update_step/grad_clip')
    total_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.clip_val)
    self.timer.end('update_step/grad_clip')

    # Update parameters with optimizer
    self.timer.start('update_step/optimizer')
    for group in self.optimizer.param_groups:
        group['lr'] = lr
    self.optimizer.step()
    self.timer.end('update_step/optimizer')
    return loss.item(), total_norm.item()


def update_target_params(self):
    """
    Update parametes of Q' with parameters of Q
    """
    self.update_target()


if __name__ == '__main__':
    env = EnvTest((5, 5, 1))
    # log=logging.getLogger('matplotlib.font_manager').disabled = True

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
                                     config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
                                 config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
