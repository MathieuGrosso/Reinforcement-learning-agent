
import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split,Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click
from torch import optim
import numpy as np 
import datetime
from icecream import ic
from utils import *
from datamaestro import prepare_dataset



class EncoderLinear(nn.Module):
    """lin´eaire → ReLU → lin´eaire, 
    la derni`ere couche produisant deux vecteurs, 
    l’un pour µ l’autre pour σ"""
    def __init__(self,dim_in,dim_hidden,dim_out):
        super(EncoderLinear,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_in,dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden,dim_out)
        )
    def forward(self,x):
      
        out = self.model(x)
   
        
        return out

class DecoderLinear(nn.Module):
    def __init__(self,dim_in,dim_hidden,dim_out):
        super(DecoderLinear,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_in,dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden,dim_out)) #dim out must be dim of x. 
        
    def forward(self,x):
        out = self.model(x)
        
   
        
        return out


class EncoderConv(nn.Module):
    """lin´eaire → ReLU → lin´eaire, 
    la derni`ere couche produisant deux vecteurs, 
    l’un pour µ l’autre pour σ"""
    def __init__(self,nchannels,hchannels):
        super(EncoderConv,self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(nchannels,hchannels,5,2),
            nn.ReLU(),
            nn.Conv2d(hchannels,hchannels*2,5,2),
            nn.ReLU()
        )

    def forward(self,x):

        x = x.view(x.shape[0],1,28,28)

        out = self.model(x)
        out = out.view(-1,out.shape[1]*out.shape[2]*out.shape[3])
   
        
        return out

class DecoderConv(nn.Module):
    """lineaire → ReLU → lineaire, 
    la derniere couche produisant deux vecteurs, 
    l’un pour µ l’autre pour σ"""
    def __init__(self,nchannels,hchannels,dimin):
        super(DecoderConv,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dimin, 300),
            nn.ReLU(),
            nn.Linear(300, 4*4*32),
            nn.ReLU()
        )

        self.model = nn.Sequential(
           nn.ConvTranspose2d(hchannels*2,hchannels,5,2),
            nn.ReLU(),
            nn.ConvTranspose2d(hchannels,nchannels,5,2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4)
        )

       
    def forward(self,x):
     
        out = self.fc(x)
        out = out.view(-1,32,4,4)
        out = self.model(out)

        out = out.view(-1,28*28)
       

   
        
        return out


        





