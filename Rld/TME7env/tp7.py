"""" Fichier à lancer pour obtenir les résultats"""


import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from utils import *
from icecream import ic
from nets import *
from memory import *
#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
batch_size   = 32
buffer_limit = 50000
tau          = 0.005 # for target network soft update

writer = SummaryWriter("XP"+time.asctime())

# class ReplayBuffer():
#     def __init__(self, buffer_limit = 50000):
#         self.buffer = collections.deque(maxlen=buffer_limit)

#     def put(self, transition):
#         self.buffer.append(transition)
    
#     def sample(self, n):
#         mini_batch = random.sample(self.buffer, n)
#         s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

#         for transition in mini_batch:
#             s, a, r, s_prime, done = transition
#             s_lst.append(s)
#             a_lst.append([a])
#             r_lst.append([r])
#             s_prime_lst.append(s_prime)
#             done_mask = 0.0 if done else 1.0 
#             done_mask_lst.append([done_mask])
        
#         return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
#                 torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
#                 torch.tensor(done_mask_lst, dtype=torch.float)
    
#     def size(self):
#         return len(self.buffer)

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
      
def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s,a,r,s_prime,done_mask  = memory.sample(batch_size)
    
    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.smooth_l1_loss(q(s,a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s,mu(s)).mean() # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()
    
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
    
def main():
    env_name  ='Pendulum-v1'
    algo_name = 'DDPG'
    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y-%HH%M-%SS")
    env = gym.make(env_name)
    memory = ReplayBuffer()
    outdir = "./XP/" + env_name  + "/" + algo_name + "_" + date_time
    logger = LogMe(SummaryWriter(outdir))
    

    dim_s, = env.observation_space.shape
    dim_a, = env.action_space.shape
    
    q, q_target = QNet(dim_s,dim_a), QNet(dim_s,dim_a)
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(dim_s,dim_a), MuNet(dim_s,dim_a)
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 20

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer  = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    for n_epi in range(10000):
        s = env.reset()
        done = False
        
        while not done:
            a = mu(torch.from_numpy(s).float()) 
            a = a.item() + ou_noise()[0]
            # a = a.detach()
            # ic(a)
            s_prime, r, done, info = env.step([a])
            memory.put((s,a,r/100.0,s_prime,done))
            score +=r
            s = s_prime
                
        if memory.size()>2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q,  q_target)
        
        if n_epi%print_interval==0 and n_epi!=0:
            ic(n_epi)
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            logger.direct_write("avg reward", score/print_interval , n_epi)
            score = 0.0
            loadTensorBoard(outdir)
    env.close() 


main()
