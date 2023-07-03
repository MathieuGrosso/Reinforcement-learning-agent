import numpy as np
from icecream import ic
import torch 
import collections
from typing import List 
import random

class ReplayBuffer:
    
    
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, s: List[np.array], a: List[np.array], r: List[float], s_prime: List[np.array], done: List[bool]):
        self.buffer.append((s, a, r, s_prime, done))

    def sample(self, N: int):
        mini_batch = random.sample(self.buffer, N)
        n_agents = len(mini_batch[0][0])  # number of states in the first transition of the mini-batch

        s_lst, a_lst, r_lst, s_prime_lst, done_lst = (
            [[] for i in range(n_agents)], [[] for i in range(n_agents)], [[] for i in range(n_agents)],
            [[] for i in range(n_agents)], [[] for i in range(n_agents)]
        )

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            for i in range(n_agents):
                s_lst[i].append(s[i])
                a_lst[i].append(a[i])
                r_lst[i].append([r[i]])
                s_prime_lst[i].append(s_prime[i])
                done_lst[i].append([done[i]])
        # s = [torch.tensor(s_lst[i], dtype=torch.float) for i in range(n_agents)]
        # a = [torch.tensor(a_lst[i], dtype=torch.float) for i in range(n_agents)]
        # r = [torch.tensor(r_lst[i], dtype=torch.float) for i in range(n_agents)]
        # s_prime = [torch.tensor(s_prime_lst[i], dtype=torch.float) for i in range(n_agents)]
        # done =  [torch.tensor(done_lst[i], dtype=torch.float) for i in range(n_agents)]   
        
        return ([torch.tensor(s_lst[i], dtype=torch.float) for i in range(n_agents)],  # n_agents * (N, dim_s)
                [torch.tensor(a_lst[i], dtype=torch.float) for i in range(n_agents)],  # n_agents * (N, dim_a)
                [torch.tensor(r_lst[i], dtype=torch.float) for i in range(n_agents)],  # n_agents * (N, 1)
                [torch.tensor(s_prime_lst[i], dtype=torch.float) for i in range(n_agents)],  # n_agents * (N, dim_s)
                [torch.tensor(done_lst[i], dtype=torch.float) for i in range(n_agents)]) # n_agents * (N, 1)

        
    def __len__(self):
        return len(self.buffer)