import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
#matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from torch import optim
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import yaml
from datetime import datetime



import random
from icecream import ic
from torch.distributions import Categorical
from icecream import ic
from nets import *
from memory import *



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
        if done:
            return True
        else:
            return False

class PPOKL(Agent):
    def __init__(self, env, opt, logger):
        super(PPOKL, self).__init__(env, opt)
        self.nSteps = 0
        self.gamma  = opt.gamma
        self.epochs = opt.epochs 
        self.logger = logger 
        self.memory = PPOMemory(mem_size=opt.mem_size, gamma=opt.gamma, tau = opt.tau, batch_size=opt.batch_size)
        self.beta_k = opt.beta_k
        self.delta  = opt.delta
        self.kl_div = torch.nn.KLDivLoss(reduction = 'none', log_target = True)
        self.Actor  = Actor(self.featureExtractor.outSize,env.action_space.n,layers = [2])
        self.Critic = Critic(self.featureExtractor.outSize,layers=[2])
        self.optimizer_actor = optim.Adam(self.Actor.parameters(), lr=opt.lr_actor, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.optimizer_critic = optim.Adam(self.Critic.parameters(), lr=opt.lr_critic, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = torch.nn.HuberLoss()
        # self.entropy = torch.nn.BCELoss()
        self.reverse = opt.reverse

    def get_distrib(self,obs):
        probs = self.Actor(obs)
        return Categorical(probs)

    def t(self,x):
        return torch.tensor(x,dtype=torch.float)

    
    def act(self,obs):
        obs = self.t(obs)
        value = self.Critic(obs)
        probs = self.get_distrib(obs)
        action = probs.sample()
        logprobs = probs.log_prob(action)

        tr = {"ob" : obs,
              "action": action,
              "log_prob": logprobs,
              "value": value}

        if not self.test:
            self.memory.store(tr)
        return action.item()

    def store(self,ob, action, new_ob, reward, done, it, episode_count):
        self.episode_count = episode_count
        if not self.test:
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            edit_tr = {"reward": reward,
                       "new_ob": new_ob,
                       "done": done,
                       "step": episode_count}
            self.memory.edit_last_transition(**edit_tr)
    
            
    def timeToLearn(self,done):
        """
            Collect set of trajectories Dk by running current policy for freqOptim steps
        """
        if self.test:
            return False
        self.nbEvents+=1
        self.nSteps+=1
        if self.nSteps%self.opt.freqOptim == 0:
            return True
    
    def learn(self):
        self.memory.compute_gae()
        for _ in range(self.epochs):
            for obs, old_probs, actions, advantage, value in self.memory.generate_batches():
                distrib = self.get_distrib(obs)
                new_critic_value = self.Critic(obs)

                #entropy = distrib.entropy().mean()
                new_probs = distrib.log_prob(actions)

                if self.reverse: 
                    kl_loss = self.kl_div(new_probs,old_probs)
                else: 
                    kl_loss = self.kl_div(old_probs,new_probs)
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage * prob_ratio
                actor_loss = -( weighted_probs - self.beta_k * kl_loss).mean()
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()

                # Critic loss

                returns = advantage + value
                # critic_loss = self.entropy(returns,new_critic_value)
                critic_loss = self.criterion(returns,new_critic_value)
                # critic_loss = (returns - new_critic_value).pow(2).mean()
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()
                
            
            # ic(kl_loss.mean())

            if kl_loss.mean() >= 1.5 * self.delta:
                self.beta_k *= 2
            else:
                if kl_loss.mean() <= self.delta / 1.5:
                    self.beta_k *= 0.5
            # ic(self.kl_adapt)
        self.memory.clear()

class PPOclipped(PPOKL):
    def __init__(self, env, opt, logger):
        super(PPOclipped, self).__init__(env, opt,logger)
        self.eps_clipped = opt.eps_clipped
        self.logger = logger
        self.criterion = torch.nn.HuberLoss()

    def learn(self):
        self.memory.compute_gae()
        for _ in range(self.epochs):
            for obs, old_probs, actions, advantage, value in self.memory.generate_batches():
                distrib = self.get_distrib(obs)
                new_critic_value = self.Critic(obs)

                #entropy = distrib.entropy().mean()
                new_probs = distrib.log_prob(actions)

                
               

                prob_ratio = (new_probs - old_probs).exp() #exp de log prob pour faire la division. 
                weighted_probs = advantage * prob_ratio
                clamped_prob_ratio = torch.clamp(prob_ratio,1-self.eps_clipped,1+self.eps_clipped)
                clamped_weighted_ratio = advantage * clamped_prob_ratio

                actor_loss = -torch.min(weighted_probs,clamped_weighted_ratio).mean()
                
                
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()

                # Critic loss

                returns = advantage + value
                critic_loss = self.criterion(returns, new_critic_value)
                # critic_loss = (returns - new_critic_value).pow(2).mean()
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()
                
            
            # ic(kl_loss.mean())
            # ic(self.kl_adapt)
        self.memory.clear()


if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_random_cartpole.yaml', "PPOclipped ")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = PPOclipped(env,config, logger)


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

            agent.store(ob, action, new_ob, reward, done, j, i)
            rsum += reward

            if agent.timeToLearn(done):
                agent.learn()
            
            # if i % config["freqTarget"] == 0:
            #     agent.update_target()

            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close()