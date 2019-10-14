import sys
import gym
import time
import torch
import numpy as np
import os
import pickle
from collections import namedtuple

CUDA = torch.cuda.is_available()
print('CUDA has been enabled.' if CUDA is True else 'CUDA has been disabled.')

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


env = gym.make('Pong-v0')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    a = int(key - ord('0'))
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a


def key_release(key, mod):
    global human_agent_action
    a = int(key - ord('0'))
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0


def save_memory(memory):
    '''
    Saves memory to relative path.
    '''
    path = os.getcwd() + '/memory'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + '/' + 'human.memory', 'wb') as f:
        pickle.dump(memory, f)


def load_memory(filename='human'):
    '''
    Loads human play memory from relative path.
    '''
    with open(os.getcwd() + '/memory/' + filename + '.memory', 'rb') as f:
        return pickle.load(f)


env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release
memory = load_memory()


def preprocess(x):
    x = torch.tensor(x).permute([2, 0, 1]).data.numpy()
    x = np.mean(x[:, ::2, ::2], axis=0) / 255
    return x.reshape(-1, 1, 105, 80)


def rollout(env):
    s = preprocess(env.reset())
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        s1, r, done, info = env.step(a)
        s1 = preprocess(s1)

        if done:
            r = -1

        memory.push(Tensor(s), a, Tensor(s1), r)

        s = s1

        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done: break
        time.sleep(0.1)

    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

    print("Frames in memory:", len(memory.memory))


print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    print('Memory preloaded with', len(memory.memory), 'samples')
    window_still_open = rollout(env)
    save_memory(memory)

    if window_still_open==False: break

