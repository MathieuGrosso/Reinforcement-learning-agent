import argparse
from os import X_OK, truncate
import sys
from pathlib import Path
import matplotlib
#matplotlib.use("Qt5agg")
# matplotlib.use("TkAgg")
import gym
import gridworld
import torch
import random
import torch.nn as nn
from memory import Memory
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
# import highway_env
from matplotlib import pyplot as plt
import yaml
from icecream import ic
from datetime import datetime





class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, env, opt):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents=0


    def act(self, obs):
        a=self.action_space.sample()
        return a

    # sauvegarde du modèle
    def save(self,outputDir):
        pass

    # chargement du modèle.
    def load(self,inputDir):
        pass

    # apprentissage de l'agent. Dans cette version rien à faire
    def learn(self):
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            pass
        pass

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = (ob, action, reward, new_ob, done)
            self.lastTransition=tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        # ic(done)
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0

### mà revoir celui là 
class DQNAgent(object):
    """DQN without experience replmay (à refaire marche pas encore) """

    def __init__(self, env, opt):
        super(DQNAgent, self).__init__(env, opt)
        self.Q = NN(inSize =self.featureExtractor.outSize, outSize=self.action_space.n, layers=[100])
        self.explo = opt.explo
        self.optim = torch.optim.Adam(params= self.Q.parameters(), lr = opt.lr)
        self.lr = opt.lr
        self.decay = opt.decay
        self.discount = opt.gamma
        self.criterion = torch.nn.SmoothL1Loss()

    def act(self, obs):
        obs = torch.from_numpy(obs)
        with torch.no_grad():
            q_s = self.Q(obs)
        next_a = np.argmax(q_s, axis = 1).item()
        if random.random() <= self.explo:
            return self.action_space.sample()
        else:
            return next_a

    # apprentissage de l'agent. Dans cette version rien à faire
    def learn(self):
        # sampler les mini batch et faire la descente de gradient ici.
        # Si l'agent est en mode de test, on n'entraîne pas

        # decay at each episode
        self.explo *= self.decay

        if self.test:
            pass
        else:
            ob, action, reward, new_ob, done = self.lastTransition
            ob = torch.from_numpy(ob)
            new_ob = torch.from_numpy(new_ob)
            with torch.no_grad():
                q_hat = self.Q(new_ob)
            q = self.Q(ob)
            target = reward + self.discount * torch.max(q_hat) * (1 - done)
            loss = self.criterion(q[0, action], target)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
    def update_target(self):
        pass



##jsp pq ca marche pas
class ExpReplayAgent(object):
    def __init__(self,env,opt):
        super(ExpReplayAgent,self).__init__()
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.capacity = opt.mem_size
        self.nbEvents=0
        self.batch_size = opt.batch_size
        self.explo = opt.explo
        self.lr = opt.lr
        self.discount = opt.gamma 
        self.criterion = nn.SmoothL1Loss()

        self.Q = NN(self.featureExtractor.outSize, env.action_space.n, layers =[100])
        self.target_network = opt.target_network
        if opt.target_network :
            self.Q_target = NN(inSize =self.featureExtractor.outSize, outSize=self.action_space.n, layers=[100])
            self.Q_target.load_state_dict(self.Q.state_dict())
            self.Q_target.eval()
        self.optim = torch.optim.Adam(self.Q.parameters())

        self.experience_replay = opt.experience_replay
        if self.experience_replay=='prioritized':
            print("experience replay prioritized")
            self.memory = Memory(self.capacity,prior=True)     
        elif self.experience_replay=='not prioritized':
            self.memory = Memory(self.capacity,prior=False)
        else: 
            pass
    
        
    def act(self,ob): 
        if random.random()<self.explo: 
            return random.randint(0,self.action_space.n-1)
        else: 
            return (torch.argmax(self.Q(torch.tensor(ob,dtype=torch.float)))).item()
    

    def save(self,path):
        pass
    
    # chargement du modèle.
    def load(self,path):
        pass

    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = (ob, action, reward, new_ob, done)
            self.memory.store(tr)

    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0
        
    



    def update_target(self):
      
        if self.target_network: 
            self.Q_target.load_state_dict(self.Q.state_dict())
            print('we have updated the model')
    
    def learn(self):
        # Si l'agent est en mode de test, on n'entraîne pas
        
        if self.test or self.batch_size >= self.memory.nentities: 
            pass
        else: 
            idx,_,batch = self.memory.sample(self.batch_size)

            
            ob_batch = torch.tensor([x[0] for x in batch])
  
            action_batch =  torch.tensor([x[1] for x in batch]).unsqueeze(-1).unsqueeze(-1)
     
            reward_batch = torch.tensor([x[2] for x in batch]).unsqueeze(-1)
   
            new_ob_batch = torch.tensor([x[3] for x in batch])
            done_batch   = torch.tensor([x[4] for x in batch], dtype=torch.int).unsqueeze(-1)

            if self.target_network: 
                q_hat = self.Q_target(new_ob_batch)
            else : 
                with torch.no_grad():
                    q_hat = self.Q(new_ob_batch)

            q = self.Q(ob_batch) # prediction 
          
            
            target = reward_batch + (self.discount*q_hat.max(dim = 2).values)*(1-done_batch) #target

            
            loss = self.criterion(torch.gather(q,2 , action_batch), target.detach())
           

            loss = self.criterion(torch.gather(q,2 , action_batch), target.detach())
  
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

    

