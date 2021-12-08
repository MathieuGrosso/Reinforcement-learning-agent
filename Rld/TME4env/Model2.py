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



class DQN1(object):
    """DQN without experience replay """

    def __init__(self, env, opt,capacity):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents=0
        self.batch_size = 32
        self.explo      = 0.1

        self.capacity   = capacity
        self.action_net = NN(self.featureExtractor.outSize, env.action_space.n, [env.action_space.n//2])
        self.target_net = NN(self.featureExtractor.outSize, env.action_space.n, [env.action_space.n//2])
        self.target_net.load_state_dict(self.action_net.state_dict())
        self.target_net.eval()
        self.time_to_update = 1001
        self.discount = 0.99
        self.batch_size = 32
        self.explo      = 0.1
        self.criterion = nn.SmoothL1Loss()
        self.optim = torch.optim.Adam(self.target_net.parameters())
    
    def act(self,ob): 
        if random.random()<self.explo: 
            return random.randint(0,self.action_space.n-1)
        else: 
            return (torch.argmax(self.action_net(torch.tensor(ob,dtype=torch.float)))).numpy()
    
    # sauvegarde du modèle
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
            self.ob = ob
            self.action = action 
            self.new_ob = new_ob
            self.reward = reward
            self.done = done
           
        
    
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0
    
    # apprentissage de l'agent. Dans cette version rien à faire
    def learn(self,epoch):
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            pass
        else: 
            if epoch % self.time_to_update == 0 : 
                self.target_net.load_state_dict(self.action_net.state_dict())
                print('we have updated the model')
            y = self.action_net(torch.tensor(self.ob, dtype=torch.float))
            if self.done: 
                yhat = self.reward
            else: 
                print("yhat is defined according to target net")
                pred = self.target_net(torch.tensor(self.new_ob, dtype=torch.float))
                yhat = self.reward + self.discount *(torch.max(pred))
            loss = self.criterion(yhat,y)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

   


class DQN_expreplay(object):
    """DQN with experience replay (prioritized or not)"""
    def __init__(self, env, opt,capacity,experience_replay):
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
    
    # sauvegarde du modèle
    def save(self,path):
        pass
    
    # chargement du modèle.
    def load(self,path):
        pass

    def update_target(self):
        if self.target_network:
            self.Q_target.load_state_dict(self.Q.state_dict())
    

    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = (ob, action, reward, new_ob, done)
            self.memory.store(tr)
    
    def update_target(self):
        if self.target_network: 
            self.Q_target.load_state_dict(self.Q.state_dict())
            print('we have updated the model')
    
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0
    
    
    def learn(self,epoch):
        # Si l'agent est en mode de test, on n'entraîne pas
        
        if self.test:
            pass
        else: 

            idx,w,batch = self.memory.sample(self.batch_size)
            for i in batch[2]:
                self.ob    = i[0]
                self.action = torch.tensor(i[1])
                self.reward = torch.tensor(i[2])
                self.new_ob = i[3]
                self.bool = i[4]
               
                if self.bool: 
                    yhat = self.reward
                else: 
        
                    pred = self.target_net(torch.tensor(self.new_ob, dtype=torch.float))
                    yhat = self.reward + self.discount *(torch.max(pred))
                y = self.action_net(torch.tensor(self.ob, dtype=torch.float))
                loss = self.criterion(yhat,y)
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

   
