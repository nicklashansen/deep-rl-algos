# Deep Reinforcement Learning with PyTorch

### Recommended prerequisites

If you're unfamiliar with deep reinforcement learning, check out this survey for a quick overview: [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/abs/1708.05866). If you're unfamiliar with reinforcement learning in general, refer to this free book on the subject: [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book.html) by Richard S. Sutton & Andrew G. Barto.

### Algorithms

This repository contains implementations of the following deep RL algorithms: ```REINFORCE``` (policy gradient), ```DQN``` (deep Q-network), ```DDQN``` (dualing deep Q-networks) and ```Actor-Critic``` (a combination of the two). Each algorithm has been implemented in a separate executible python script that can be run directly from the terminal:

```
python REINFORCE.py --env Pong-v0 --num_episodes 10000
```

If you inspect the scripts, you will find a number of other hyper-parameters that can also be set by arguments, such as discount factor (gamma) and learning rate.

### Results
Experiments have been conducted on the ```CartPole-v0```, ```LunarLander-v2```, ```Pong-v0``` and ```BreakoutDeterministic-v4``` environments. CartPole-v0 and LunarLander-v2 are solved in a few minutes while Pong-v0 takes roughly 16 hours on a single thread equipped with a Tesla V100. BreakoutDeterministic-v4 remains unsolved but the repository contains code that should be able to solve the environment eventually given enough time.
Below is a few GIFs showcasing REINFORCE and DQN on LunarLander-v2.

**REINFORCE vs. LunarLander-v2**

![reinforce-lunarlander](https://i.imgur.com/Q7GioKq.gif)

**DQN vs. LunarLander-v2**

![dqn-lunarlander](https://i.imgur.com/4FGK7X5.gif)

### Where to go from here?

If you want to dive deeper into my work in deep reinforcement learning, refer to my [implementation of A3C](https://github.com/nicklashansen/a3c) and my [minimal neural architecture search system](https://github.com/nicklashansen/minimal-nas).
