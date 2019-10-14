import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
from collections import namedtuple
import random
from gym import wrappers
import os
import pickle

CUDA = torch.cuda.is_available()
print('CUDA has been enabled.' if CUDA is True else 'CUDA has been disabled.')

BATCH_SIZE = 32
COMPLETE_SIZE = 10
FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
IntTensor   = torch.cuda.IntTensor if CUDA else torch.IntTensor
LongTensor  = torch.cuda.LongTensor if CUDA else torch.LongTensor
ByteTensor  = torch.cuda.ByteTensor if CUDA else torch.ByteTensor
Tensor      = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent(object):
    def __init__(self,
                 policy=None,
                 critic=None,
                 env=None,
                 num_episodes=1000,
                 discount_factor=0.99,
                 lr=3e-4,
                 test_freq=200,
                 test_num=10,
                 min_reward=-250,
                 max_reward=3000000,
                 conv=True,
                 name = "un-named"):
        super(Agent, self).__init__()

        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.lr = lr
        self.test_freq = test_freq
        self.test_num = test_num
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.achieved_max_reward = False
        #self.rollout_limit = env.spec.timestep_limit
        self.conv = conv
        if self.conv:
            self.rollout_limit = 10000
        else:
            self.rollout_limit = 10000

        self.name = name
        if env is not None: self.env = env

        if policy is not None:
            self.policy = policy.cuda() if CUDA else policy
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
            if critic is not None:
                self.critic = critic.cuda() if CUDA else critic
                self.optimizerC = optim.Adam(self.critic.parameters(), lr=lr)

        self.test_n, self.test_r = [], []
        self.losses = []


    def reset_env(self, env=None):
        """
        Resets the current environment using a constant
        seed to make sure environment is deterministic.
        """
        if env is None: env = self.env
        env.seed(0)

        if self.conv:
            return self.preprocess(env.reset())
        return env.reset()


    def select_action(self, s):
        """
        Selects an action according to the current policy.
        """
        s = Variable(Tensor(s))
        action_logits = self.policy(s)
        log_probs = action_logits-torch.logsumexp(action_logits, dim = 1)

        action = torch.distributions.Categorical(logits=action_logits).sample()

        return action.data.cpu().numpy(), log_probs[0,action.data.cpu().numpy()]



    def transform_reward(self, r):
        return np.sign(r)


    def take_action(self, state):
        if self.conv:
            state = self.preprocess(state)
        action = self.select_action(state)
        return action[0]


    # def preprocess(self, x):
    #     x = torch.tensor(x).permute([2, 0, 1]).data.numpy()
    #     x = np.mean(x[:, ::2, ::2], axis=0) / 255
    #     return x.reshape(-1, 1, 105, 80)


    def preprocess(self, x):
        x = torch.tensor(x).permute([2, 0, 1]).data.numpy()
        x = np.mean(x[:, ::2, ::2], axis=0) / 255
        x = x[17:105-8]
        new_frame = x.reshape(-1, 1, 80, 80)
        if not hasattr(self, "old_frame"): self.old_frame = new_frame
        diff_frame = new_frame - self.old_frame
        self.old_frame = new_frame
        return diff_frame


    def play_episode(self, env=None, replay=False):
        """
        Plays a single episode and returns SAR rollout.
        The logarithm of the action probabilities is also
        included in the rollout (for computing the loss).
        """
        train = env is None
        if train:
            env = self.env

        s = self.reset_env(env)
        rollout, eps_r = [], 0

        for i in range(self.rollout_limit):
            a, log_probs = self.select_action(s)
            s1, r, done, _ = env.step(a)

            if self.conv is True:
                s1 = self.preprocess(s1)
                # r = self.transform_reward(r)

            rollout.append((s, a, r, log_probs))
            eps_r += r

            if train:
                if self.conv is True and self.epsilon > self.epsilon_min:
                    self.epsilon -= (self.epsilon_max - self.epsilon_min) / self.epsilon_steps
                if hasattr(self, 'memory'):
                    self.memory.push(Tensor(s), a, Tensor(s1), r)
                if replay: self.replay()
                if eps_r < self.min_reward and env is None: break
            if done: break

            s = s1

        if eps_r > self.max_reward:
            print('Achieved maximum reward:', eps_r)
            self.achieved_max_reward = True

        return np.array(rollout)


    def compute_loss(self, rewards, log_probs):
        """
        Computes the loss from discounted return.
        """
        G, loss = torch.zeros(1,1).type(FloatTensor), 0
        #rewards= (rewards-np.mean(rewards))/(np.std(rewards)+1e-05)
        for i in reversed(range(len(rewards))):
            G = self.discount_factor * G + (rewards[i])
            loss = loss - (log_probs[i]*Variable(G))
        return loss


    def train(self):
        """
        Runs a full training for defined number of episodes.
        """
        complete_array = np.zeros(COMPLETE_SIZE)
        for e in range(1, self.num_episodes+1):
            rollout = self.play_episode()
            self.optimize(rollout)

            if self.conv is False and self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if e % self.test_freq == 0:
                n, r = self.test()
                print('{:5d},  Reward: {:6.2f},  Length: {:4.2f}'.format(e, r, n))
                complete_array[(e//10)%10] = r
            if self.achieved_max_reward: break

        print('Completed training!')
        #self.plot_rewards()


    def test(self):
        """
        Runs a number of tests and computes the
        mean episode length and mean reward.
        """
        n, r = [], []

        for e in range(self.test_num):
            rollout = self.play_episode()
            rewards = np.array(rollout[:, 2], dtype=float)
            n.append(len(rollout))
            r.append(sum(rewards))

        self.test_n.append(n)
        self.test_r.append(r)

        save_policy(self, self.name)

        return np.mean(n), np.mean(r)


    def get_replay(self):
        """
        Renders an episode replay using the current policy.
        """
        env = wrappers.Monitor(self.env, "./gym-results", force=True)
        state = env.reset()
        while True:
            env.render()
            action = self.take_action(state)
            state_next, reward, terminal, info = env.step(action)
            state = state_next
            if terminal: break

        env.close()


    def plot_rewards(self):
        """
        Plots the moving average of the reward during training.
        """
        def moving_average(a, n=10) :
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret / n

        plt.figure(figsize=(16,6))
        plt.subplot(211)
        plt.plot(range(1, len(self.train_r)+1), self.train_r, label='training reward')
        plt.plot(moving_average(self.train_r))
        plt.xlabel('episode'); plt.ylabel('reward')
        plt.xlim((0, len(self.train_r)))
        plt.legend(loc=4); plt.grid()
        plt.subplot(212)
        plt.plot(range(1, len(self.losses)+1), self.losses, label='loss')
        plt.plot(moving_average(self.losses))
        plt.xlabel('episode'); plt.ylabel('loss')
        plt.xlim((0, len(self.losses)))
        plt.legend(loc=4); plt.grid()
        plt.tight_layout(); plt.show()


def save_policy(agent, filename):
    '''
    Saves a policy of specified filename to relative path.
    '''
    path = os.getcwd() + '/agents'
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(agent.policy.state_dict(), path + '/' + filename + '.policy')
    save_stats(agent.test_n, agent.test_r, filename)


def load_policy(agent, filename):
    '''
    Loads a policy of specified filename to relative path.
    '''
    agent.policy.load_state_dict(torch.load( path + '/' + filename + '.policy'))


def save_agent(agent, filename, delete_memory=True):
    '''
    Saves an agent of specified filename to relative path.
    '''
    path = os.getcwd() + '/agents'
    if not os.path.exists(path):
        os.makedirs(path)
    if delete_memory and hasattr(agent, "memory"):
        agent.memory=None
    with open(path + '/' + filename + '.agent', 'wb') as f:
        pickle.dump(agent, f)
    save_stats(agent.test_n, agent.test_r, filename)


def load_agent(filename):
    '''
    Loads an agent of specified filename from relative path.
    '''
    with open(os.getcwd() + '/agents' + '/' + filename + '.agent', 'rb') as f:
        return pickle.load(f)


def save_stats(n, r, filename):
    '''
    Saves stats of specified filename to relative path.
    '''
    path = os.getcwd() + '/agents'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/' + filename + '.stats', 'wb') as f:
        pickle.dump((n, r), f)


def load_stats(filename):
    '''
    Loads stats of specified filename from relative path.
    '''
    with open(os.getcwd() + '/agents' + '/' + filename + '.stats', 'rb') as f:
        return pickle.load(f)
