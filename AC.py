import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import gym
from agent import Agent
from DQN import PolicyNet as CriticNet
from REINFORCE import PolicyNet as ActorNet

CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

class Agent(Agent):
    
    def select_action(self, s):
        """
        Selects an action according to the current policy.
        """
        s = Variable(FloatTensor(s))
        action_probs = self.policy(s)
        state_value = self.critic(s)
        log_probs = action_probs.log()
        action = torch.distributions.Categorical(action_probs).sample()
        return action.data.cpu().numpy(), log_probs[action], state_value
        
    def play_episode(self, env=None, replay=False):
        """
        Plays a single episode and returns SAR rollout.
        The logarithm of the action probabilities is also
        included in the rollout (for computing the loss).
        """
        if env is None: env = self.env
        s = self.reset_env(env)
        rollout, eps_r = [], 0
    
        for i in range(self.rollout_limit):
            a, log_probs, state_value = self.select_action(s)
            s1, r, done, _ = env.step(a)
            rollout.append((s, a, r, log_probs,state_value))
            eps_r += r
            if hasattr(self, 'memory'): self.memory.push(FloatTensor(s), a, FloatTensor(s1), r)
            if replay: self.replay()
            if eps_r < self.min_reward and env is None: break
            if done: break
            s = s1
    
        if eps_r > self.max_reward:
            print('Achieved maximum reward:', eps_r)
            self.achieved_max_reward = True
            
        return np.array(rollout)
    
    def compute_loss(self, rewards, log_probs, state_values):
        """
        Computes the loss from discounted return.
        """
        G, loss, value_loss = torch.zeros(1,1).type(FloatTensor), 0, 0
        
        for i in reversed(range(len(rewards))):
            G = self.discount_factor * G + (rewards[i])
            advantage=Variable(G)-state_values[i]
            loss = loss - (log_probs[i]*advantage)
            value_loss = value_loss + advantage*advantage#F.smooth_l1_loss(state_values[i], rewards[i]*torch.ones(1).type(FloatTensor))
        return (loss) / len(rewards), value_loss / len(rewards)

    
    
class AC(Agent):
    def __init__(self,
                 env_arg='LunarLander-v2',
                 env=gym.make('LunarLander-v2'),
                 num_episodes=10000,
                 discount_factor=0.99,
                 lr=1e-3,
                 test_freq=250,
                 test_num=10,
                 name=None):
        super(AC, self).__init__(
            env=gym.make(env_arg),
            policy=ActorNet(n_in=env.observation_space.shape[0], n_out=env.action_space.n),
            critic=CriticNet(n_in=env.observation_space.shape[0], n_out=1),
            num_episodes=num_episodes,
            discount_factor=discount_factor,
            lr=lr,
            test_freq=test_freq,
            test_num=test_num)


    def optimize(self, rollout):
        rewards = np.array(rollout[:,2], dtype=float)
        log_probs = np.array(rollout[:,3])
        state_values = np.array(rollout[:,4])
        
        self.optimizer.zero_grad()
        self.optimizerC.zero_grad()
        actor_loss, critic_loss = self.compute_loss(rewards, log_probs,state_values)
        actor_loss.backward(retain_graph=True)      
        critic_loss.backward()
        self.optimizer.step()
        self.optimizerC.step()

        self.train_r.append(sum(rewards))
        self.train_n.append(len(rollout))
        self.losses.append(actor_loss.detach().cpu())
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_arg', default='LunarLander-v2', type=str)
    parser.add_argument('--num_episodes', default=800, type=int)
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--test_freq', default=50, type=int)
    parser.add_argument('--test_num', default=10, type=int)
    parser.add_argument('--name_postfix',default='', type=str)
    args = parser.parse_args()
    agent = AC(env_arg=args.env_arg,
                 num_episodes=args.num_episodes,
                 discount_factor=args.discount_factor,
                 lr=args.lr,
                 test_freq=args.test_freq,
                 test_num=args.test_num,
                 name=(
                    "AC"
                    f"_num_episodes={args.num_episodes}"
                    f"_discount_factor={args.discount_factor}"
                    f"_lr={args.lr}"
                    f"_test_freq={args.test_freq}"
                    f"_test_num={args.test_num}"
                    f"{args.name_postfix}"
                ))
    agent.train()
    agent.get_replay()