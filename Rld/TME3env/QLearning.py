import matplotlib
matplotlib.use("TkAgg")
import gym
import random
import gridworld
# from gym import wrappers, logger
import numpy as np
import copy
from icecream import ic
from datetime import datetime
import os
from utils import *


class Agent:


    def __init__(self, env, opt):
        self.opt=opt
        self.action_space = env.action_space
        self.env=env
        self.discount=opt.gamma
        self.alpha=opt.learningRate
        self.explo=opt.explo
        self.exploMode=opt.exploMode #0: epsilon greedy, 1: ucb
        self.algorithm = opt.algorithm
        ic(self.algorithm)
        self.modelSamples=opt.nbModelSamples
        self.test=False
        self.tau = opt.tau
        self.strategy = opt.strategy
        self.qstates = {}  # dictionnaire d'états rencontrés
        self.values = []   # contient, pour chaque numéro d'état, les qvaleurs des self.action_space.n actions possibles



    def save(self,file):
       pass


    # enregistre cette observation dans la liste des états rencontrés si pas déjà présente
    # retourne l'identifiant associé à cet état
    def storeState(self,obs):
        observation = obs.dumps()
        s = str(observation)
        ss = self.qstates.get(s, -1)

        # Si l'etat jamais rencontré
        if ss < 0:
            ss = len(self.values)
            self.qstates[s] = ss
            self.values.append(np.ones(self.action_space.n) * 1.0) # Optimism faced to uncertainty (on commence avec des valeurs à 1 pour favoriser l'exploration)
        return ss



    def act(self, obs):
        if self.exploMode == 0:
            if np.random.random()<self.explo:
                return self.action_space.sample()
            else: 
                return np.argmax(self.values[obs])
        if self.exploMode==1: #Boltzmann
            probs = [np.exp(self.values[obs][a] / self.tau) / np.sum(np.exp(self.values[obs] / self.tau)) for a in range(4)]
            return np.random.choice(4, p=probs)



    def store(self, ob, action, new_ob, reward, done, it):

        if self.test:
            return
        self.ob=ob
        self.action=action
        self.new_ob=new_ob
        self.reward=reward
        if it == self.opt.maxLengthTrain:   # si on a atteint la taille limite, ce n'est pas un vrai done de l'environnement
            done = False
        self.done=done



    def learn(self, done,action_next=None):
        if self.algorithm == 'QLearning':
            self.values[self.ob][self.action] += self.alpha* (self.reward + self.discount * np.max(self.values[self.new_ob])- self.values[self.ob][self.action])
        if self.algorithm == 'Sarsa':
            # ic('we do sarsa')
            self.values[self.ob][self.action] += self.alpha *(self.reward+self.discount*self.values[self.new_ob][action_next]- self.values[self.ob][self.action])


# #not working
# class DynaQ(Agent):
#     def __init__(self,env,opt):
#         super(DynaQ,self).__init__(env,opt)
#         # estimates of R(s_t, a_t, s_t+1)
#         self.R = defaultdict(lambda: [defaultdict(lambda: 0)] * 4)
#         # and P(s_t+1 | s_t, a_t)
#         self.P = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
#         self.alpha_R = opt.alpha_r
#         self.k = opt.k
        

#     def learn(self,done):
#         #update value: 
#         self.values[self.ob][self.action] += self.alpha* (self.reward + self.discount * np.max(self.values[self.new_ob])- self.values[self.ob][self.action])
        
#         #update MDP : 
#         self.R[self.ob][self.action][self.new_ob] = self.reward # set the value once and for all because the reward is deterministic in our case
#         self.P[self.new_ob][self.ob][self.action] += self.alpha_R * (1 - self.P[self.new_ob][self.ob][self.action])
#         for s_other in self.P.keys():
#             self.P[s_other][self.ob][self.action] += self.alpha_R * (0 - self.P[s_other][self.ob][self.action])

#         # Sample k state-action tuples, and update the Q value functions given the new MDP estimate
        
#         for s_i in np.random.choice(list(self.values[self.ob].item()), min(len(self.values), self.k), replace=False):
#             a_i = np.random.choice(4, replace=False)
#             self.values[s_i][a_i] += self.alpha * (sum([
#                 self.P[s_other][s_i][a_i] * (self.R[s_i][a_i][s_other] + self.gamma * np.max(self.values[s_other]))
#                 for s_other in self.P.keys()
#             ]) - self.values[s_i][a_i])






if __name__ == '__main__':
    env,config,outdir,logger=init('./configs/config_qlearning_gridworld.yaml',"Sarsa")
    
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]


    agent = Agent(env,config)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    nb = 0
    for i in range(episode_count):
        checkConfUpdate(outdir, config)  # permet de changer la config en cours de run

        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()

        if (i > 0 and i % int(config["freqVerbose"]) == 0):
            verbose = True
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:  ##### Si agent.test alors retirer l'exploration
            print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()
        new_ob = agent.storeState(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action = agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.storeState(new_ob)

            j+=1

            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                #print("forced done!")

            agent.store(ob, action, new_ob, reward, done, j)
            if agent.algorithm=='Sarsa':
                action_next = agent.act(new_ob)
                agent.learn(done,action_next)
            else: 
                agent.learn(done)
         
            rsum += reward
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                break


    loadTensorBoard(outdir)     
         
    env.close()