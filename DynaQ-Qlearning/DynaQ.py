
import matplotlib
matplotlib.use("TkAgg")
import gym
import gridworld
# from gym import wrappers, logger
import numpy as np
import copy
from datetime import datetime
import os
from utils import *
from icecream import ic
import random
import math 



class DynaQ(object):

    def __init__(self, env, opt):
        self.opt=opt
        self.action_space = env.action_space
        self.env=env
        self.discount=opt.gamma
        self.decay = opt.decay
        self.alpha=opt.learningRate
        self.explo=opt.explo
        self.exploMode=opt.exploMode #0: epsilon greedy, 1: ucb
        self.sarsa=opt.sarsa
        self.modelSamples=opt.nbModelSamples
        self.test=False
        self.qstates = {}  # dictionnaire d'états rencontrés
        self.values = []   # contient, pour chaque numéro d'état, les qvaleurs des self.action_space.n actions possibles
        self.R = {}
        self.P = {}
        self.tau = opt.tau
        self.taken_actions = {}   # contient, pour chaque numéro d'état, les actions prises


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
        q_s = self.values[obs]
        if self.exploMode == 0:
            if random.random() < self.explo:
                next_a = random.randint(0,self.action_space.n-1)
            else:
                next_a = np.argmax(q_s)
        if self.exploMode==1:
            probs = [np.exp(self.values[obs][a] / self.tau) / np.sum(np.exp(self.values[obs] / self.tau)) for a in range(4)]
            next_a = np.random.choice(4, p=probs)
        #else: implémenter UCB
        if obs not in self.taken_actions:
            self.taken_actions[obs] = []
        self.taken_actions[obs].append(next_a)
        return next_a


    def store(self, ob, action, new_ob, reward, done, it):
        if self.test:
            return
        self.last_source=ob
        self.last_action=action
        self.last_dest=new_ob
        self.last_reward=reward
        if it == self.opt.maxLengthTrain:   # si on a atteint la taille limite, ce n'est pas un vrai done de l'environnement
            done = False
        self.last_done=done


    def learn(self, done):
        q_s = self.values[self.last_source]
        q_s_next = self.values[self.last_dest]
        q_s[self.last_action] += self.alpha*(self.last_reward +self.discount*np.max(q_s_next)-q_s[self.last_action])

    def _update_R(self):
        if self.last_source not in self.R:
            self.R[self.last_source] = {}
        if self.last_action not in self.R[self.last_source]:
            self.R[self.last_source][self.last_action] = {}
        if self.last_dest not in self.R[self.last_source][self.last_action]:
            self.R[self.last_source][self.last_action][self.last_dest] = 0
        self.R[self.last_source][self.last_action][self.last_dest] += self.alpha*(self.last_reward-self.R[self.last_source][self.last_action][self.last_dest])
    
    def _update_P(self):
        if self.last_source not in self.P:
            self.P[self.last_source] = {}
        if self.last_action not in self.P[self.last_source]:
            self.P[self.last_source][self.last_action] = {}
        S = list(self.taken_actions.keys()).copy()
        if self.last_dest in list(self.taken_actions.keys()):
            S.remove(self.last_dest)
        for s_prime in S:
            if s_prime not in self.P[self.last_source][self.last_action]:
                self.P[self.last_source][self.last_action][s_prime] = 1/self.action_space.n #Quelles valeurs prendre pour initialiser self.P ?
            else:
                self.P[self.last_source][self.last_action][s_prime] += -self.alpha*(self.P[self.last_source][self.last_action][s_prime])
        if self.last_dest not in self.P[self.last_source][self.last_action]:
            self.P[self.last_source][self.last_action][self.last_dest] = 1/self.action_space.n
        self.P[self.last_source][self.last_action][self.last_dest] += self.alpha*(1-self.P[self.last_source][self.last_action][self.last_dest])
        

    def update_model(self):
        self._update_R()
        self._update_P()

    def _learn_planning(self, state, action):
            S = list(self.taken_actions.keys())
            q_s = self.values[state]
            Sum = 0
            for s_prime in S:
                if state in self.R:
                    if action in self.R[state]:
                        if s_prime in self.R[state][action]:
                            Sum += self.P[state][action][s_prime]*(self.R[state][action][s_prime]+self.discount*np.max(self.values[s_prime])-q_s[action])
            q_s[action] += self.alpha*Sum

    def planning_step(self):
        for k in range(self.modelSamples):
            state = random.choice(list(self.taken_actions.keys()))
            action = random.choice(self.taken_actions[state])
            self._learn_planning(state, action)


if __name__ == '__main__':
    env,config,outdir,logger=init('./configs/config_qlearning_gridworld.yaml',"DynaQ-plan5")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    nbModeSamples = config["nbModelSamples"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

   
    agent = DynaQ(env, config)


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
        i+=1
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

            if agent.sarsa:
                next_action = agent.act(new_ob)
            agent.learn(done)

            if nbModeSamples > 0:
               
                agent.update_model()
                agent.planning_step()

            rsum += reward
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                logger.direct_write('mean reward',mean,i)
                
                break
        
        agent.explo = agent.explo*agent.decay
    loadTensorBoard(outdir)


    env.close()