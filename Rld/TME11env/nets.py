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
from icecream import ic
from typing import List 
from core import *
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import yaml
from datetime import datetime

from memory import * 


class Mu(nn.Module): 
    def __init__(self, config,dim_state): 
        super(Mu,self).__init__()
        self.dim_state = dim_state 
        ic(config.h,config.action_size)
        self.ff = nn.Sequential(
            nn.BatchNorm1d(dim_state),
            nn.Linear(dim_state,2*config.h),
            nn.ReLU(),
            nn.Linear(2*config.h,config.h),
            nn.ReLU(),
            nn.Linear(config.h,config.action_size)
            )
        self.clipped = config.action_high
    def forward(self,x):
        return self.clipped*torch.tanh(self.ff(x))

class Q(nn.Module):
    def __init__(self, config,nb_agent): 
        super(Q,self).__init__()
        dim_state = config.dim_state 
        h = config.h 
        ic(dim_state)
        ic(nb_agent)
        self.ff = nn.Sequential(
            nn.BatchNorm1d(sum(dim_state)+config.action_size*nb_agent),
            nn.Linear(sum(dim_state)+config.action_size*nb_agent,2*h),
            nn.ReLU(),
            nn.Linear(2*h,h),
            nn.ReLU(),
            nn.Linear(h,1)
            )
    def forward(self,s,a_list):
        x = torch.cat(s+a_list,dim=-1)
        return self.ff(x)


class QNet(nn.Module):
    def __init__(self, dim_s: List[int], dim_a: List[int], hidden_sizes: List[int]):
        super(QNet, self).__init__()
        sizes = [sum(dim_s) + sum(dim_a)] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.LeakyReLU()]
        layers += [nn.Linear(sizes[-1], 1)]  # last layer
        self.model = nn.Sequential(*layers)

    def forward(self, s: List[torch.tensor], a: List[torch.tensor]):
        # a: n_agents * (N, dim_a_i)
        # s: n_agents * (N, dim_s_i)
        a_cat = torch.cat(a, dim=1)
        s_cat = torch.cat(s, dim=1)
        return self.model(torch.cat([s_cat, a_cat], dim=1))


class MuNet(nn.Module):
    def __init__(self, dim_s: int, dim_a_i: int, hidden_sizes: List[int]):
        super(MuNet, self).__init__()
        sizes = [dim_s] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]),  # hidden layers
                       nn.LeakyReLU()]
        layers += [nn.Linear(sizes[-1], dim_a_i),  # last layer
                   nn.Tanh()]  # tanh to normalize output
        self.model = nn.Sequential(*layers)

    def forward(self, s):
        return self.model(s)

