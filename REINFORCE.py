import torch.nn.functional as F
import torch.nn as nn
import random
import torch
import numpy as np
import gym
from torch.autograd import Variable
from collections import namedtuple
from agent import Agent, ReplayMemory, BATCH_SIZE, CUDA, save_agent
from conv import PolicyConvNet

FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
IntTensor   = torch.cuda.IntTensor if CUDA else torch.IntTensor
LongTensor  = torch.cuda.LongTensor if CUDA else torch.LongTensor
ByteTensor  = torch.cuda.ByteTensor if CUDA else torch.ByteTensor
Tensor      = FloatTensor

class PolicyNet(nn.Module):

    def __init__(self,
                 n_hidden1=256,
                 n_hidden2=128,
                 n_in=8,
                 n_out=4):
        super(PolicyNet, self).__init__()
        self.n_in=n_in
        self.n_out=n_out
        self.fc1 = nn.Linear(self.n_in, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.out = nn.Linear(n_hidden2, self.n_out)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)



class REINFORCE(Agent):
    def __init__(self,
                 env_arg='LunarLander-v2',
                 num_episodes=100,
                 discount_factor=0.99,
                 lr=1e-3,
                 test_freq=250,
                 test_num=10,
                 conv = True,
                 name=None):
        env = gym.make(env_arg)
        self.outputsize = 4 # env.action_space.n
        super(REINFORCE, self).__init__(
            env=env,
            policy=PolicyConvNet(n_out = env.action_space.n) if conv else PolicyNet(n_in=env.observation_space.shape[0], n_out=self.outputsize),
            num_episodes=num_episodes,
            discount_factor=discount_factor,
            lr=lr,
            test_freq=test_freq,
            test_num=test_num,
            conv = conv,
            name = name)
        print("playing game " + env_arg)


    def optimize(self, rollout):
        rewards = np.array(rollout[:,2], dtype=float)
        log_probs = np.array(rollout[:,3])

        self.optimizer.zero_grad()
        loss = self.compute_loss(rewards, log_probs)
        loss.backward()
        self.optimizer.step()



    def train(self):
        """
        Runs a full training for defined number of episodes.
        """
        completed_array = np.zeros(COMPLETE_SIZE)

        for e in range(1, self.num_episodes+1):
            rollout = self.play_episode()
            self.optimize(rollout)

            if e % self.test_freq == 0:
                n, r = self.test()
                print('{:5d},  Reward: {:6.2f},  Length: {:4.2f}'.format(e, r, n))

                if not self.conv:
                    completed_array[(e//COMPLETE_SIZE)%COMPLETE_SIZE] = r

            if np.mean(completed_array) >= MAX_SCORE: break

        print('Completed training!')

    def play_episode(self, env=None, replay=False):
        """
        Plays a single episode and returns SAR rollout.
        The logarithm of the action probabilities is also
        included in the rollout (for computing the loss).
        """
        if env is None: env = self.env
        s = self.reset_env(env)
        rollout, eps_r, mapo_traj = [], 0, []
       # s = self.preprocess(s)
        for i in range(self.rollout_limit):
            a, log_probs = self.select_action(s)
            s1, r, done, _ = env.step(a)
            if self.conv:
                s1 = self.preprocess(s1)
            rollout.append((s, a, r, log_probs))
            if done: r += -1
            eps_r += r
            if hasattr(self, 'memory'):
                self.memory.push(Tensor(s), a, Tensor(s1), r)
            if replay: self.replay()
            if eps_r < self.min_reward and env is None: break
            if done: break
            s = s1

        if eps_r > self.max_reward:
            #print('Achieved maximum reward:', eps_r)
            self.achieved_max_reward = True

        return np.array(rollout)


    def select_action(self, s):
        """
        Selects an action according to the current policy.
        """
        s = Variable(Tensor(s))
        action_logits = self.policy(s)
        if self.conv:
            log_probs = action_logits-torch.logsumexp(action_logits, dim = 1)
        else:
            log_probs = action_logits-torch.logsumexp(action_logits, dim = 0)

        action = torch.distributions.Categorical(logits=action_logits).sample()

        if self.conv:
            return action.data.cpu().numpy(), log_probs[0,action.data.cpu().numpy()]

        return action.data.cpu().numpy(), log_probs[action]


    def compute_loss(self, rewards, log_probs):
        """
        Computes the loss from discounted return.
        """
        G, loss = torch.zeros(1,1).type(FloatTensor), 0

        #if not self.conv:
        #    rewards= (rewards-np.mean(rewards))/(np.std(rewards)+1e-05)

        for i in reversed(range(len(rewards))):
            G = self.discount_factor * G + (rewards[i])
            loss = loss - (log_probs[i]*Variable(G))

        return loss


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




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Pong-v0', type=str) # 'LunarLander-v2', 'BreakoutDeterministic-v4'
    parser.add_argument('--num_episodes', default=10000, type=int)
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--test_freq', default=50, type=int)
    parser.add_argument('--test_num', default=10, type=int)
    parser.add_argument('--name_postfix',default='', type=str)
    #parser.add_argument('--score',default=200, type=int)
    #parser.add_argument('--conv', default=0, type=int)
    args = parser.parse_args()

    if args.env == 'BreakoutDeterministic-v4':
        MAX_SCORE = 1000
    elif args.env == 'Pong-v0':
        MAX_SCORE = 21
    elif args.env == 'LunarLander-v2':
        MAX_SCORE = 200
    else:
        MAX_SCORE = 1e9


    if args.env in ['BreakoutDeterministic-v4','Pong-v0']:
        conv = True
    else:
        conv = False

    COMPLETE_SIZE = 10

    agent = REINFORCE(env_arg=args.env,
                 #env=gym.make(args.env_arg),
                 num_episodes=args.num_episodes,
                 discount_factor=args.discount_factor,
                 lr=args.lr,
                 test_freq=args.test_freq,
                 test_num=args.test_num,
                 conv=conv,#not args.env=='LunarLander-v2',
                 name=(
                    "REINFORCE"
                    f"_env={args.env}"
                    f"_num_episodes={args.num_episodes}"
                    f"_discount_factor={args.discount_factor}"
                    f"_lr={args.lr}"
                    f"_test_freq={args.test_freq}"
                    f"{args.name_postfix}"
                ))
    agent.train()
    # agent.get_replay()
