import torch
from DQN import PolicyNet,DQN
from collections import namedtuple
from agent import  BATCH_SIZE, CUDA
from conv import PolicyNet

FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
IntTensor   = torch.cuda.IntTensor if CUDA else torch.IntTensor
LongTensor  = torch.cuda.LongTensor if CUDA else torch.LongTensor
ByteTensor  = torch.cuda.ByteTensor if CUDA else torch.ByteTensor
Tensor      = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(DDQN, self).__init__(*args,**kwargs)
        
        self.target = PolicyNet()
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        
        if CUDA:
            self.target = self.target.cuda()

    
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), dtype=torch.uint8)
        non_final_next_states = torch.cat([Tensor(s) for s in batch.next_state
                                                    if s is not None]).reshape(BATCH_SIZE, -1)

        state_batch = torch.cat(batch.state).reshape(BATCH_SIZE, -1)
        action_batch = torch.cat(torch.split(LongTensor(batch.action), 1)).reshape(BATCH_SIZE, -1)
        reward_batch = Tensor(batch.reward)

        state_action_values = self.policy(state_batch).gather(1, action_batch)
        target_next_state_values = self.target(non_final_next_states)
        
        next_state_values = Tensor(torch.zeros(BATCH_SIZE))
        next_state_values[non_final_mask] = self.policy(non_final_next_states).gather(1, torch.max(target_next_state_values, 1)[1].unsqueeze(1)).squeeze(1).detach()
        expected_state_action_values = ((next_state_values * self.discount_factor) + reward_batch)
        
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    def train(self):
        """
        Runs a full training for defined number of episodes.
        """
        for e in range(1, self.num_episodes+1):
            rollout = self.play_episode(replay=True)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            if e % self.update_target == 0:
                self.target.load_state_dict(self.policy.state_dict())

            if e % self.test_freq == 0:
                n, r = self.test()
                print('{:5d},  Reward: {:6.2f},  Length: {:4.2f}, Epsilon: {:4.2f}'.format(e, r, n, self.epsilon))
                
            if self.achieved_max_reward: break

        print('Completed training!')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_arg', default='LunarLander-v2', type=str)
    parser.add_argument('--num_episodes', default=3000, type=int)
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--test_freq', default=50, type=int)
    parser.add_argument('--test_num', default=10, type=int)
    parser.add_argument('--update_target', default=10, type=int)
    parser.add_argument('--epsilon', default=1.0, type=float)
    parser.add_argument('--epsilon_decay', default=0.9993, type=float)
    parser.add_argument('--epsilon_min', default=0.01, type=float)
    parser.add_argument('--name_postfix',default='', type=str)
    parser.add_argument('--conv', default=0, type=int)
    args = parser.parse_args()
    agent = DDQN(env_arg=args.env_arg,
                 num_episodes=args.num_episodes,
                 discount_factor=args.discount_factor,
                 lr=args.lr,
                 test_freq=args.test_freq,
                 test_num=args.test_num,
                 update_target = args.update_target,
                 epsilon = args.epsilon,
                 epsilon_decay = args.epsilon_decay,
                 epsilon_min = args.epsilon_min,
                 conv = True if args.conv == 1 else False,
                 name=(
                    "DDQN"
                    f"_num_episodes={args.num_episodes}"
                    f"_discount_factor={args.discount_factor}"
                    f"_lr={args.lr}"
                    f"_test_freq={args.test_freq}"
                    f"_test_num={args.test_num}"
                    f"_update_target={args.update_target}"
                    f"_epsilon={args.epsilon}"
                    f"_epsilon_decay={args.epsilon_decay}"
                    f"_epsilon_min={args.epsilon_min}"
                    f"{args.name_postfix}"
                ))
    agent.train()
    agent.get_replay()