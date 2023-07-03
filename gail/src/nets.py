import datetime
import glob
import pickle
import shutil
from pathlib import Path
import numpy as np
import gym
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from memory import Memory


class Pi(nn.Module):
    def __init__(self,dim_s,dim_a):
        super(Pi,self).__init__()
        self.pi = nn.Sequential(
            nn.Linear(dim_s, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, dim_a)
        )
        
    def forward(self,x):
        return self.pi(x)