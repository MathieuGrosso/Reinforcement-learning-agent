import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
#matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
from core import NN
import random
from memory import Memory
from icecream import ic

class Agent(object):
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
        # sampler les mini batch et faire la descente de gradient ici.
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
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0

class DQNAgent(Agent):
    """The world's simplest agent!"""

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

class ExpReplayAgent(Agent):
    def __init__(self, env, opt):
        super(ExpReplayAgent, self).__init__(env, opt)
        self.explo = opt.explo
        self.lr = opt.lr
        self.decay = opt.decay
        self.discount = opt.gamma
        self.featureExtractor = opt.featExtractor(env)
        self.Q = NN(inSize =self.featureExtractor.outSize, outSize=self.action_space.n, layers=[100])
        self.optim = torch.optim.Adam(params= self.Q.parameters(), lr = opt.lr)
        self.prior = opt.prior
        self.memoire = Memory(mem_size = opt.mem_size, prior= self.prior)
        self.batch_size = opt.batch_size
        self.target_network = opt.target_network
        if opt.target_network :
            self.Q_target = NN(inSize =self.featureExtractor.outSize, outSize=self.action_space.n, layers=[100])
            self.Q_target.load_state_dict(self.Q.state_dict())
            self.Q_target.eval()
        self.criterion = torch.nn.SmoothL1Loss(reduction='none')
     # enregistrement de la transition pour exploitation par learn ulterieure


    def act(self, obs):
        obs = torch.tensor(obs)
        with torch.no_grad():
            q_s = self.Q(obs)
            next_a = torch.argmax(q_s).item()
        if random.random() <= self.explo:
            return self.action_space.sample()
        else:
            return next_a

    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = (ob, action, reward, new_ob, done)
            idx = self.memoire.store(tr)

    def learn(self):
        # sampler les mini batch et faire la descente de gradient ici.
        # Si l'agent est en mode de test, on n'entraîne pas
        # decay at each episode
        # self.explo *= self.decay
        if self.test or self.batch_size >= self.memoire.nentities:
            return
        else:
            idx, w, batch = self.memoire.sample(self.batch_size)
            ic(w)
            obs_batch, a_batch, r_batch, next_obs_batch, done_batch = \
                                        torch.tensor([x[0] for x in batch]), \
                                        torch.tensor([x[1] for x in batch]).unsqueeze(-1).unsqueeze(-1), \
                                        torch.tensor([x[2] for x in batch]).unsqueeze(-1), \
                                        torch.tensor([x[3] for x in batch]), \
                                        torch.tensor([x[4] for x in batch], dtype=torch.int).unsqueeze(-1)
                                        
            w = torch.tensor(w).detach()
            if self.target_network:
                q_hat = self.Q_target(next_obs_batch)
            else:       
                with torch.no_grad():
                    q_hat = self.Q(next_obs_batch)
    
            q = self.Q(obs_batch)
            target = r_batch + self.discount * q_hat.max(dim = 2).values * (1 - done_batch)
            if self.prior:
                loss = self.criterion(torch.gather(q,2 , a_batch).squeeze(-1), target.detach()) * w
            else:
                loss = self.criterion(torch.gather(q,2 , a_batch).squeeze(-1), target.detach())
            loss = loss.mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if self.prior:
                with torch.no_grad():   
                    tderror = (target - torch.gather(q,2 , a_batch).squeeze(-1)).detach().numpy() 
                    self.memoire.update(idx, tderror)

    def update_target(self):
        if self.target_network:
            self.Q_target.load_state_dict(self.Q.state_dict())
    

if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_random_cartpole.yaml', "RandomAgent")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = ExpReplayAgent(env,config)


    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        new_ob = agent.featureExtractor.getFeatures(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action= agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j+=1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob, action, new_ob, reward, done,j)
            rsum += reward

            if agent.timeToLearn(done):
                agent.learn()
            
            if i % config["freqTarget"] == 0:
                ic(i)
                agent.update_target()

            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close()