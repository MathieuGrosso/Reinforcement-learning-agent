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

from memory import * 


class MuNet(nn.Module):
    def __init__(self,dimIn,dimA):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(dimIn, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, dimA)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))*2 # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        return mu

class QNet(nn.Module):
    def __init__(self,dimIn,dimA):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(dimIn, 64)
        self.fc_a = nn.Linear(dimA,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32,1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q


class CriticNetwork(nn.Module):
    def __init__(self,inSize,dim_a,outSize):
        super(CriticNetwork, self).__init__()
        # inSize = dim_in+dim_a
        self.fc_s = nn.Linear(inSize, 64)
        self.fc_a = nn.Linear(dim_a,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32,1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=2)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q


class Actor(nn.Module):
    def __init__(self,inSize,dim_a):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(inSize,128)
        self.fc2 = nn.Linear(128,64)
        self.fc_mu = nn.Linear(64,1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))*2 # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        return mu

class Critic(nn.Module):
    def __init__(self,inSize,dim_a):
        super(Critic, self).__init__()
        self.fc_s = nn.Linear(inSize, 64)
        self.fc_a = nn.Linear(dim_a,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32,1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=2)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q


class ActorNetwork(nn.Module):
    def __init__(self, inSize, outSize) -> None:
        super(ActorNetwork, self).__init__()
        # ic(inSize)
        self.fc1 = nn.Linear(inSize, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)


    def forward(self, x):
        # ic(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))*2 # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        return mu