import torch.nn.functional as F
import torch.nn as nn
import random
import torch
import numpy as np
import gym
from collections import namedtuple
from agent import Agent, ReplayMemory, BATCH_SIZE, CUDA

class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, padding=False):
        super(Conv, self).__init__()

        padding = int((kernel_size - 1) / 2) if padding else 0
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=stride,
                      padding=padding),
            #nn.BatchNorm2d(out_channels),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class PolicyConvNet(nn.Module):

    def __init__(self, n_out):
        super(PolicyConvNet, self).__init__()

        self.conv1 = Conv(1, 32, kernel_size=8, stride=4)
        self.conv2 = Conv(32, 64, kernel_size=4, stride=2)
        self.conv3 = Conv(64, 64, kernel_size=3)

        self.fc1 = nn.Linear(2304, 512)
        self.out = nn.Linear(512, n_out)

    def forward(self, x):

        x = x.reshape(-1, 1, 80, 80)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)
		
