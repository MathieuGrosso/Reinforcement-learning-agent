import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
#matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from torch import optim
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import yaml
from datetime import datetime



import random

from torch.distributions import Categorical
from icecream import ic

from memory import *
class feedforwardNet(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):

        super(feedforwardNet, self).__init__()
        self.layers = nn.ModuleList([])
        ic(layers)
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))

    def setcuda(self, device):
        self.cuda(device=device)

    def forward(self, x):

        x = self.layers[0](x)
        for i in range(1, len(self.layers)):

            x = torch.tanh(x)
            x = self.layers[i](x)
            
        return x

class Actor(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(Actor, self).__init__()
        self.net = feedforwardNet(inSize, outSize, layers)

    def forward(self, x):
        output = self.net(x)
        out = (F.softmax(output))
        return out

class Critic(nn.Module):
    def __init__(self, inSize, layers=[]):
        super(Critic, self).__init__()
        self.net = feedforwardNet(inSize, 1, layers)

    def forward(self, x):
        output = self.net(x)
        return output

class ActorNetwork(nn.Module):
    def __init__(self, inSize, outSize) -> None:
        super(ActorNetwork, self).__init__()
        self .fc1 = nn.Linear(inSize, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, outSize)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

class CriticNetwork(nn.Module):
    def __init__(self, inSize, outSize) -> None:
        super(CriticNetwork, self).__init__()
        self .fc1 = nn.Linear(inSize, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128,  1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
