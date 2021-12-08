import torch
import random
import torch.nn as nn
from torch.distributions import Categorical

from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
# import highway_env
from matplotlib import pyplot as plt
import yaml
from icecream import ic
from datetime import datetime
from core import NN
import torch.nn.functional as f

class Actor(nn.Module):
    """decides which action should be taken """
    def __init__(self,inSize,outSize,layers=[]):
        super(Actor,self).__init__()
        self.net = NN(inSize,outSize,layers)
    def forward(self,x):
        out=self.net(x)
        distributions= f.softmax(out)
        output = Categorical(distributions)
        return output

class Critic(nn.Module):
    """Inform the actor how good was the action and how it should adjust"""
    def __init__(self,inSize,layers=[]):
        super(Critic,self).__init__()
        self.net = NN(inSize,1,layers) #output is size 1 becaue give juste an information. 
    def forward(self,x):
        return self.net(x)




class Batch_ActorCritic(object): 
    def __init__(self, env, opt,logger,mode,test_mode,layer):
            self.opt=opt
            if opt.fromFile is not None:
                self.load(opt.fromFile)
            self.env=env
            # self.state_dim = state_dim
            # self.action_dim=action_dim
            self.time_to_update = 10001
            self.logger = logger
            self.test = test_mode
            self.mode = mode
            self.action_space = env.action_space
            self.featureExtractor = opt.featExtractor(env)
            self.layer=layer

            self.Actor = Actor(self.featureExtractor.outSize,env.action_space.n,self.layer)
            self.Critic1 = Critic(self.featureExtractor.outSize,self.layer) # nombre d'état possible en entrée et autant d'action que possible en sortie
            self.Critic2 = Critic(self.featureExtractor.outSize,self.layer) #
            self.Critic2.load_state_dict(self.Critic1.state_dict())
            self.Critic2.eval()



            self.criterion = torch.nn.HuberLoss()
            self.lr=0.01
            self.optim_Actor = torch.optim.Adam(self.Actor.parameters(),lr=self.lr)
            self.optim_Critic = torch.optim.Adam(self.Critic2.parameters(),lr=self.lr)
            self.discount = 0.99
            self.t = 0 
            self.batch_size = 32
            self.transition = []

    
    def save(self,path):
        pass
    
        

    # chargement du modèle.
    def load(self,path):
        pass
            
    
    def act(self,obs):
        
        ob=(self.featureExtractor.getFeatures(obs))
        m = self.Actor(torch.tensor(ob,dtype=torch.float)) #use actor to predict the action
        action = m.sample()
        ic(action)

        # distributions = f.softmax(self.policy_net((torch.tensor(ob,dtype=torch.float).unsqueeze(0)))) # policy net définit l'esperance de reward de chaque action dans un état # a voir si on prend policy net ou target net
        # m=Categorical(distributions) # on prend le meilleur choix
        
        # ic(action[0][0])
        
        log_prob = m.log_prob(action)
        new_ob, reward, done, _ = self.env.step(action[0].numpy())
        new_ob = self.featureExtractor.getFeatures(new_ob)
        liste=[]
        liste.append(log_prob)
        liste.append(ob)
        liste.append(done)
        liste.append(reward)
        liste.append(new_ob)
        self.transition.append(liste)

        return new_ob,reward,done
    
    def update(self,epoch):
        if epoch % self.time_to_update == 0 :

            self.Critic1.load_state_dict(self.Critic2.state_dict())
            ic('we have updated the model')
    
    def restart_transition(self,done):
        self.transition=[]
        return self.transition

    def timeToLearn(self,done):
        if self.test:
            return False
        else : 
            if done : 
                self.nbEvents+=1
            return done 
    
    def learn(self):
        logprob=[]

        advantage=[]
        
        for idx,batch in enumerate(self.transition): #batch
            if self.mode == "TD0":
                log_prob=batch[0]
                ob=batch[1]
                done=batch[2]
                reward=batch[3]
                new_ob=batch[4]
                prediction = self.Critic2(torch.tensor(ob,dtype=torch.float))
                ic(prediction)
               
                value=reward+self.discount*self.Critic1(torch.tensor(new_ob,dtype=torch.float))
                advantage.append(value - prediction)
                logprob.append(log_prob)


            if self.mode == "MC":
                pass

        ic(advantage)
        ic(logprob)
        advantage=torch.cat([i for i in advantage])
        logprob=torch.cat([i for i in logprob])
        
        lossJ=logprob*advantage.detach() #on veut le gradient que de log prob
        lossJ=-lossJ.mean() # car on veut l'ajouter et pas le soustraire au moment de la backprop
        torch.autograd.set_detect_anomaly(True)
        self.optim_Actor.zero_grad()
        lossJ.backward() # on update les parametres de la policy (actor) apres avoir eu l'avis de la critic. 
        self.optim_Actor.step()

        loss = self.criterion(prediction,value)
        self.optim_Critic.zero_grad()
        loss.backward()
        self.optim_Critic.step()  # on update la critic. 
     

        self.logger.direct_write('loss_per_episode', loss.item(), self.t)
        self.logger.direct_write('lossJ_per_episode ', lossJ.item(), self.t)
        self.t += 1

        









        

