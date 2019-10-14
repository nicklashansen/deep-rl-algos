import torch.nn.functional as F
import torch.nn as nn
import random
import torch
import numpy as np
import gym
from collections import namedtuple
from agent import *
from conv import PolicyConvNet

FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
IntTensor   = torch.cuda.IntTensor if CUDA else torch.IntTensor
LongTensor  = torch.cuda.LongTensor if CUDA else torch.LongTensor
ByteTensor  = torch.cuda.ByteTensor if CUDA else torch.ByteTensor
Tensor      = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class PolicyNet(nn.Module):

    def __init__(self,
                 n_hidden1=64,
                 n_hidden2=64,
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


class DQN(Agent):
    def __init__(self,
                 env_arg='LunarLander-v2',
                 num_episodes=2000,
                 discount_factor=0.99,
                 lr=2.5e-4,
                 test_freq=50,
                 test_num=10,
                 update_target=10,
                 epsilon_max=1,
                 epsilon_min=0.1,
                 epsilon_steps=1000000//8,
                 epsilon_decay=0.9993,
                 conv=True,
                 name=None):
        env = gym.make(env_arg)
        self.inputsize = env.observation_space.shape[0]
        self.outputsize = 4 # env.action_space.n
        super(DQN, self).__init__(
            env=env,
            policy=PolicyConvNet(n_out=self.outputsize) if conv else PolicyNet(n_out=env.action_space.n),
            num_episodes=num_episodes,
            discount_factor=discount_factor,
            lr=lr,
            test_freq=test_freq,
            test_num=test_num,
            conv=conv,
            name = name)
        self.conv = conv

        if CUDA:
            self.policy = self.policy.cuda()

        self.target = PolicyConvNet(n_out=self.outputsize) if conv else PolicyNet(n_out=env.action_space.n)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        if CUDA:
            self.target = self.target.cuda()

        self.loss = nn.MSELoss()
        self.memory = ReplayMemory(100000)
        self.update_target = update_target
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_steps = epsilon_steps
        self.epsilon_decay = epsilon_decay


    def select_action(self, s):
        """
        Selects an action according to the current policy.
        """
        if random.random() > self.epsilon:
            with torch.no_grad():
                return int(self.policy(Tensor(s)).argmax().cpu().data.numpy()), None
        else:
            return np.random.randint(self.outputsize), None


    def update_weights(self, transitions):
        """
        Updates the network's weights according to a
        number of given transitions.
        """
        minibatch_size = len(transitions)

        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.uint8)

        non_final_next_states = torch.cat([Tensor(s) for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(torch.split(LongTensor(batch.action), 1)).reshape(minibatch_size, -1)
        reward_batch = Tensor(batch.reward)

        if self.conv is False:
            state_batch = state_batch.reshape(minibatch_size, -1)
            non_final_next_states = non_final_next_states.reshape(minibatch_size, -1)

        state_action_values = self.policy(state_batch).gather(1, action_batch)

        next_state_values = Tensor(torch.zeros(minibatch_size).cuda() if CUDA else torch.zeros(minibatch_size))

        next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = ((next_state_values * self.discount_factor) + reward_batch)

        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def replay(self):
        """
        Samples a number of transitions from memory and replays
        the experiencies. Does not do anything until memory contains
        enough samples for a full minibatch.
        """
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        self.update_weights(transitions)


    def fill_memory(self, episodes=300):
        """
        Fills memory with random episodes.
        """
        env = self.env
        s = self.reset_env(env)

        for episode in range(episodes):
            for i in range(self.rollout_limit):
                a = np.random.randint(self.outputsize)
                s1, r, done, _ = env.step(a)
                s1 = self.preprocess(s1)

                if done:
                    r = -1

                self.memory.push(Tensor(s), a, Tensor(s1), r)
                if done: break
                s = s1


    def train(self):
        """
        Runs a full training for a defined number of episodes.
        """
        print('Filling memory...')
        self.fill_memory(100)

        print('Started training...')
        for e in range(1, self.num_episodes + 1):

            self.play_episode(replay=True)

            if self.conv is False and self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if e % self.update_target == 0:
                self.target.load_state_dict(self.policy.state_dict())
                if CUDA:
                    self.target = self.target.cuda()

            if e % self.test_freq == 0:
                n, r = self.test()
                print('{:5d},  Reward: {:6.2f},  Length: {:4.2f}, Epsilon: {:4.2f}'.format(e, r, n, self.epsilon))

            if self.achieved_max_reward: break

        print('Completed training!')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='PongNoFrameskip-v4', type=str) # 'LunarLander-v2', 'BreakoutDeterministic-v4'
    parser.add_argument('--num_episodes', default=5000, type=int)
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--lr', default=2.5e-4, type=float)
    parser.add_argument('--test_freq', default=50, type=int)
    parser.add_argument('--test_num', default=5, type=int)
    parser.add_argument('--update_target', default=10, type=int)
    parser.add_argument('--epsilon_max', default=1.0, type=float)
    parser.add_argument('--epsilon_min', default=0.1, type=float)
    parser.add_argument('--epsilon_steps', default=1000000, type=float)
    parser.add_argument('--epsilon_decay', default=0.9993, type=float)
    parser.add_argument('--name_postfix',default='', type=str)
    args = parser.parse_args()
    agent = DQN(env_arg=args.env,
                 num_episodes=args.num_episodes,
                 discount_factor=args.discount_factor,
                 lr=args.lr,
                 test_freq=args.test_freq,
                 test_num=args.test_num,
                 epsilon_max=args.epsilon_max,
                 epsilon_min=args.epsilon_min,
                 epsilon_steps=args.epsilon_steps,
                 epsilon_decay=args.epsilon_decay,
                 conv=not args.env=='LunarLander-v2',
                 name=(
                    "DQN"
                    f"_num_episodes={args.num_episodes}"
                    f"_discount_factor={args.discount_factor}"
                    f"_lr={args.lr}"
                    f"_test_freq={args.test_freq}"
                    f"{args.name_postfix}"
                ))
    agent.train()
    # agent.get_replay()
