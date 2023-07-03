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


class Memory:
    def __init__(self, batch_size,min_clip,max_clip):
        self.memory = []
        self.cumulated_r = []
        self.batch_size = batch_size
        self.min_clip  = min_clip 
        self.max_clip = max_clip

    def store(self, transition):
        self.memory.append(transition)

    def get_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        indices = range(len(self.memory))
        for i in indices:
            s, a, r, s_prime, done, info = self.memory[i]
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done])

        N, dim_a = len(a_lst), len(a_lst[0])
        return (torch.tensor(s_lst, dtype=torch.float),  # (N, dim_s)
                torch.tensor(a_lst).reshape(N, dim_a),  # (N, dim_a) even if dim_a=1 thx to reshape
                torch.tensor(r_lst, dtype=torch.float),  # (N, 1)
                torch.tensor(s_prime_lst, dtype=torch.float),  # (N, dim_s)
                torch.tensor(done_lst, dtype=torch.float))  # (N, 1)

    def get_minibatch_proxy_reward(self):
        assert len(self.cumulated_r) == len(self.memory)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        for i in indices:
            s, a, r, s_prime, done, info = self.memory[i]
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([self.cumulated_r[i]])
            s_prime_lst.append(s_prime)
            done_lst.append([done])

        N, dim_a = len(a_lst), len(a_lst[0])
        return (torch.tensor(s_lst, dtype=torch.float),  # (N, dim_s)
                torch.tensor(a_lst).reshape(N, dim_a),  # (N, dim_a) even if dim_a=1 thx to reshape
                torch.tensor(r_lst, dtype=torch.float),  # (N, 1)
                torch.tensor(s_prime_lst, dtype=torch.float),  # (N, dim_s)
                torch.tensor(done_lst, dtype=torch.float))  # (N, 1)

    def compute_cumulated_r(self, agent, dim_a):
        cumulated_rewards = []
        cumulated_rewards_real = []
        r_cumulated = 0
        r_cumulated_real = 0
        remaining_ep_len = 0

        # get GAIL rewards (using the discriminator) instead of real rewards
        obs, act, _, _, _ = self.get_batch()
        act_one_hot = F.one_hot(act.flatten(), num_classes=dim_a)
        input_agent = torch.cat([obs, act_one_hot], dim=1)
        d_agent = torch.sigmoid(agent.D(input_agent))
        rewards_clipped = torch.clamp(torch.log(d_agent), min=self.min_clip, max=self.max_clip)
        rewards_clipped = torch.flatten(rewards_clipped).tolist()

        for i in reversed(range(len(self.memory))):
            s, a, r, s_prime, done, info = self.memory[i]
            if done:
                remaining_ep_len = 0
                r_cumulated = 0
                r_cumulated_real = 0
            r_cumulated_real = r + r_cumulated_real  # R_t = sum_{t'=t}^T r_t'
            r_cumulated = rewards_clipped[i] + r_cumulated  # R_t = sum_{t'=t}^T r_t'
            remaining_ep_len += 1  # T - t
            cumulated_rewards.append(r_cumulated / remaining_ep_len)
            cumulated_rewards_real.append(r_cumulated_real / remaining_ep_len)
        self.cumulated_r = list(reversed(cumulated_rewards))

    def __len__(self):
        return len(self.memory)

