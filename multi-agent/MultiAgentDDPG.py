import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
import random
import multiagent
import multiagent.scenarios
import multiagent.scenarios.simple_tag as simple_tag
import multiagent.scenarios.simple_tag as simple_spread
import multiagent.scenarios.simple_tag as simple_adversary
from multiagent.environment import MultiAgentEnv
from torch.utils.tensorboard import SummaryWriter
import multiagent.scenarios as scenarios
from gym import wrappers, logger
import numpy as np
from torch import optim, unsqueeze 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import time
import logging
import json
import subprocess
from collections import namedtuple,defaultdict
from utils import *
from core import *
from memory import ReplayBuffer
from datetime import datetime
from nets import QNet,MuNet, Mu, Q 
from collections import deque



class Agent(object):
    floatTensor = torch.FloatTensor(1)
    longTensor = torch.LongTensor(1)

    # agent_type: 0 = random, 1 = DDPG, 2 = MADDPG
    def __init__(self,env,opt,id,action_size,obs_shapes,agent_type = 2, nb_subPolicies = 1, optimizer = torch.optim.Adam,lr = 0.1, lrq = 0.1,hidden_sizes_q=None,
                     hidden_sizes_mu=None,dimIn=None,dim_a=None,dimOut=None):


        self.id=id
        self.opt=opt

        self.eps_std = opt.sigma_noise
        self.agent_type = agent_type
        self.obs_shapes = obs_shapes
        self.env = env

        print(id,nb_subPolicies)

        self.action_size=action_size
        self.nb_subPolicies=nb_subPolicies
        self.policies = []
        self.targetPolicies = []
        self.q = None
        self.qtarget = None

        #on définit des lisstes d'agent pour MADDPG cf au dessus. 
        if agent_type==0: #random
            self.nb_subPolicies = 0
        else: 
            self.nb_subPolicies = 1
            # self.policies, self.targetPolicies = Mu(opt, dim_state), Mu(opt, dim_state)
            # self.q,self.qtarget = Q(opt,env.n), Q(opt,env.n)
		    # self.q, self.qtarget = Q(config), Q(config)
            self.q = QNet(dimIn, dim_a, hidden_sizes_q)
            self.qtarget = QNet(dimIn, dim_a, hidden_sizes_q)
            self.policies = MuNet(dimIn[id], dim_a[id], hidden_sizes_mu)
            self.targetPolicies = MuNet(dimIn[id], dim_a[id], hidden_sizes_mu)
  
            # self.nb_subPolicies = nb_subPolicies
            # self.policies = MuNet(dimIn[id], dim_a[id], hidden_sizes_mu) for i in range(nb_subPolicies)]
            # self.targetPolicies = [MuNet(dimIn[id], dim_a[id], hidden_sizes_mu) for i in range(nb_subPolicies)]
            # self.q = QNet(dimIn, dim_a, hidden_sizes_q)
            # self.qtarget = QNet(dimIn, dim_a, hidden_sizes_q)
        
            self.polyakP = self.opt.polyakP
            self.polyakQ = self.opt.polyakQ
            

        self.currentPolicy=0

        self.buffer = ReplayBuffer(buffer_limit=1000)

        # self.events = [deque(maxlen=self.opt.capacity) for i in range(nb_subPolicies)]
        self.batchsize = self.opt.batchsize
     

        if self.opt.fromFile is not None:
            self.load(self.opt.fromFile)


        # Creation optimiseurs
        if agent_type > 0:  # not random
            wdq = self.opt.wdq   # weight decay for q
            wdp = self.opt.wdp  # weight decay for pi
            self.qtarget.load_state_dict(self.q.state_dict())
            self.targetPolicies.load_state_dict(self.policies.state_dict())
            self.optimizerQ =  optimizer(params=self.q.parameters(), weight_decay=wdq, lr=lrq)
            self.optimizerP =  optimizer(params=self.policies.parameters(), weight_decay=wdp, lr=lr)
            # for i in range(nb_subPolicies):
            #     self.targetPolicies[i].load_state_dict(self.policies[i].state_dict())

            # self.optimizerP = [optimizer([{"params": self.policies[i].parameters()}], weight_decay=wdp, lr=lr) for i in
            #                        range(nb_subPolicies)]
            # self.optimizerQ = optimizer([{"params": self.q.parameters()}], weight_decay=wdq, lr=lrq)



    def act(self,obs):
        if self.agent_type == 0:
            a = self.floatTensor.new((np.random.rand(self.action_size) - 0.5) * 2).view(-1)
            return a
        else: 
            with torch.no_grad():
                a = self.policies(obs)
                eps = torch.normal(mean=0, std=self.eps_std, size=a.shape)
        return torch.clamp(a + eps, min=-1., max=1.)
       

    def getTargetAct(self,obs):
        if self.agent_type == 0:
            a=self.floatTensor.new((np.random.rand(obs.shape[0],self.action_size) - 0.5) * 2).view(-1,self.action_size)
            return a
        i = np.random.randint(0, self.nb_subPolicies, 1)[0]
        return self.targetPolicies[i](obs)
    
    def addEvent(self,ob, action, rewards, new_ob, d):
         self.buffer.put(ob, action, rewards, new_ob, d)

    def sample(self,N):
        s,a,r,s_prime,done=self.buffer.sample(N)
        return s,a,r,s_prime,done
  

    def selectPolicy(self):
        if self.agent_type==0:
            return 0
        i = np.random.randint(0, self.nb_subPolicies, 1)[0]
        self.currentPolicy = i

    def eval(self):
        if self.q is not None:
            # for p in self.policies:
            #     p.eval()
            self.policies.eval()
            self.q.eval()
    def train(self):
        if self.q is not None:
            # for p in self.policies:
            #     p.train()
            self.policies.train()
            self.q.train()

    def soft_update(self):
        if self.q is not None:

            # for i in range(len(self.policies)):
            #     for target, src in zip(self.targetPolicies[i].parameters(), self.policies[i].parameters()):
            #         target.data.copy_(target.data * self.polyakP + src.data * (1-self.polyakP))
            for target, src in zip(self.targetPolicies.parameters(), self.policies.parameters()):
                target.data.copy_(target.data * self.polyakP + src.data * (1 - self.polyakP))

            for target, src in zip(self.qtarget.parameters(), self.q.parameters()):
                target.data.copy_(target.data * self.polyakQ + src.data * (1 - self.polyakQ))

    def setcuda(self,device):
        Agent.floatTensor = torch.cuda.FloatTensor(1, device=device)
        Agent.longTensor = torch.cuda.LongTensor(1, device=device)
        if self.q is not None:
            # for p in self.policies:
            #     p.setcuda(device)
            # for p in self.targetPolicies:
            #     p.setcuda(device)
            self.targetPolicies.setcuda(device)
            self.policies.setcuda(device)
            self.q.setcuda(device)
            self.qtarget.setcuda(device)

    def save(self,file):

        if self.q is not None:
            # for x in range(self.nb_subPolicies):
            #     torch.save(self.policies[x].state_dict(), file + "_policy_" + str(self.id) + "_" + str(x) + ".txt")
            torch.save(self.q.state_dict(), file + "_value_"+str(self.id)+".txt")
            torch.save(self.policies.state_dict(), file + "_policy_" + str(self.id) + ".txt")

    def load(self,file):
        if self.q is not None:
            # for x in range(self.nb_subPolicies):
            #     self.policies[x].load_state_dict(torch.load(file +"_policy_"+str(self.id)+"_"+str(x)+".txt"))
                self.q.load_state_dict(torch.load(file + "_value_"+str(self.id)+".txt"))
                self.policies.load_state_dict(torch.load(file + "_policy_"+str(self.id)+".txt"))


class MADDPG(object):
    floatTensor = torch.FloatTensor(1)
    longTensor = torch.LongTensor(1)
    verbose = 0
    def __init__(self,env,opt,action_size,obs_shapes,noise,noiseTest,dimIn,dim_a,dimOut, hidden_sizes_q,
                     hidden_sizes_mu):
        super(MADDPG, self).__init__()
        self.action_size = action_size
        self.env=env
        self.opt=opt
        self.gamma=self.opt.gamma
        self.maxReward = opt.maxReward
        self.minReward = -opt.maxReward

        agent_types = self.opt.agent_types
        nb_subPolicies =  self.opt.nb_subPolicies
        lr = self.opt.lr
        lrq = self.opt.lrq
        device = self.opt.device


        nbSteps = self.opt.nbSteps
        freqOptim = self.opt.freqOptim
        optimizer=torch.optim.Adam
        seed=self.opt.seed

        
        self.freqOptim=freqOptim
        self.nbSteps=nbSteps
        self.noise=noise
        self.noiseTest = noiseTest
        self.startEvents=self.opt.startEvents
        self.test=False

        self.nbEvts=0
        self.nbEvents=0

        
        self.polyakP=self.opt.polyakP
        self.polyakQ = self.opt.polyakQ
        self.nbOptim=0

        self.nbRuns=0

        #définir les agents
       
        self.agents=[]
        for i in range(env.n):
            ic(env.n)
            ic(agent_types[i])
            
            a=Agent(env,opt,i,action_size,obs_shapes,agent_types[i],nb_subPolicies[i],optimizer,lr=lr[i],lrq=lrq[i], hidden_sizes_q=hidden_sizes_q,
                     hidden_sizes_mu=hidden_sizes_mu,dimIn=dimIn,dim_a=dim_a,dimOut=dimOut)
            self.agents.append(a)


        self.nb=0

        self.sumDiff=0
        prs("lr",lr)


        self.current=[]
        self.batchsize=self.opt.batchsize

        if self.opt.fromFile is not None:
            self.load(self.opt.fromFile)

        if device>=0:
            cudnn.benchmark = True
            torch.cuda.device(device)
            torch.cuda.manual_seed(seed)
            MADDPG.floatTensor = torch.cuda.FloatTensor(1, device=device)
            MADDPG.longTensor = torch.cuda.LongTensor(1, device=device)
            for i in range(env.n):
                self.agents[i].setcuda(device)

  




    def save(self,file):
        for agent in self.agents:
            agent.save(file)

    def load(self,file):
        for agent in self.agents:
            agent.load(file)


    def store(self,ob,action,new_ob,rewards,done,it):
        # d=done[0]
        # if it == self.opt.maxLengthTrain:
        #     print("undone")
        #     d = False

        for a in self.agents:
            #print(("ob", ob, "a", action, "r", rewards, "new_ob", new_ob, d))
            a.addEvent(ob, action, rewards, new_ob, done)
        self.nbEvts += 1



    def act(self, obs):
        for a in self.agents:
            a.eval()
        if self.nbEvts>self.startEvents:
            with torch.no_grad():

                actions = torch.cat([self.agents[i].act(self.floatTensor.new(obs[i])).view(-1) for i in range(self.env.n)],dim=-1).cpu()
                #print(actions)
                #print
                if (not self.test) or self.opt.sigma_noiseTest>0:
                    noise=self.noise
                    if self.test:
                        noise=self.noiseTest
                    e=torch.cat([x.sample() for x in noise],dim=-1)
                    actions=actions+e.view(-1)
                actions=actions.numpy()

        else:
            actions =np.concatenate([x.sample().numpy() for x in self.noise],axis=-1)


        return actions.reshape((self.env.n,-1))


    def endsEpisode(self):
        for i in range(self.env.n):
            self.noise[i].reset()
            self.noiseTest[i].reset()
            self.agents[i].selectPolicy()

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    def timeToLearn(self, done):
        if self.test:
            return False
        self.nbEvents += 1
        #print(self.nbEvents,self.opt.freqOptim,self.opt.startEvents)
        if self.nbEvents % self.opt.freqOptim == 0 and  self.nbEvents > self.opt.startEvents:
            print("Time to Learn! ")
            return True
        return False

    def learn(self):
        sl=torch.nn.MSELoss()
        cl, al = [], []
        for i, agent_i in enumerate(self.agents):
            agent_i.train()
            if len(agent_i.buffer) >= self.opt.buffer_min_len :
                batch = agent_i.sample(N=self.batchsize)
                s, a, r, s_p, done = batch
                #clamp reward =
                # for i in range(len(r)): 
                #     r[i] = torch.clamp(r[i],self.minReward,self.maxReward)
                a_p = [ agent.targetPolicies(s_p[j]) for j, agent in enumerate(self.agents) ]

                #critic : 
                y = r[i].unsqueeze(-1) + self.gamma * agent_i.qtarget(s_p, a_p)
                q = agent_i.q(s,a)
                critic_loss = sl(q,y.detach())
                critic_loss = F.smooth_l1_loss(q,y.detach())
                agent_i.optimizerQ.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent_i.q.parameters(), 0.5)
                agent_i.optimizerQ.step()
                

                #actor :
                mu = [agent.policies(s[j]) for j, agent in enumerate(self.agents)]
                actor_loss = - agent_i.q(s,mu).mean(dim=0)
                actor_loss += (mu[i]**2).mean() * 1e-3
                agent_i.optimizerP.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent_i.policies.parameters(), 0.5)
                agent_i.optimizerP.step()
			  
                cl.append(critic_loss.item())
                al.append(actor_loss.item())
                agent_i.eval()


        for x in range(self.env.n):
            self.agents[x].soft_update()

