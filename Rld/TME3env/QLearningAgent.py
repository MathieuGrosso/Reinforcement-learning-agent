import matplotlib
matplotlib.use("TkAgg")
import gym
import tensorflow as tf
print(tf.__version__)
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from datetime import datetime
import os
import tkinter
from utils import *
from icecream  import ic
import random

import argparse
parser = argparse.ArgumentParser(prog='Qlearning, Sarsa parser')
parser.add_argument("--Agent_Mode",help="allow to choose the mode of the agent", action="store")
parser.add_argument("--Plan",help="allow to chose the plan",action="store")
args=parser.parse_args()


class QLearning(object):
    def __init__(self, env, opt):
        self.opt=opt
        self.action_space = env.action_space
        self.env=env
        self.discount=opt.gamma
        self.alpha=opt.learningRate
        self.epsilon=0.1
        self.explo=opt.explo
        self.exploMode=opt.exploMode #0: epsilon greedy, 1: ucb
        self.sarsa=opt.sarsa
        self.modelSamples=opt.nbModelSamples
        self.test=False
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
        action=0
        if self.exploMode==0:
            """epsilon greedy mode"""
            proba=random.random()
            if random.uniform(0,1) < self.explo : 
                action = env.action_space.sample()
                # ic(action)
                return action 
            else :
                action = np.argmax(self.values[obs])
                # ic(action)
                return action 
        else: 
            pass
     


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
        predict=self.last_reward+self.discount*np.max(self.values[self.last_dest])
        target=self.values[self.last_source][self.last_action]
        # ic(self.values[self.last_source])
        self.values[self.last_source][self.last_action]=self.values[self.last_source][self.last_action]+self.alpha*(predict-target)
        # ic(self.last_source)
        # ic(self.last_action)
        # ic(self.values[self.last_source])

        return  self.values

class Sarsa(object): 
    def __init__(self, env, opt):
        self.opt=opt
        self.action_space = env.action_space
        self.env=env
        self.discount=opt.gamma
        self.alpha=opt.learningRate
        self.epsilon=0.1
        self.explo=opt.explo
        self.exploMode=opt.exploMode #0: epsilon greedy, 1: ucb
        self.sarsa=opt.sarsa
        self.modelSamples=opt.nbModelSamples
        self.test=False
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
        action=0
        if self.exploMode==0:
            """epsilon greedy mode"""
            proba=random.random()
            if random.uniform(0,1) < self.explo : 
                action = env.action_space.sample()
                # ic(action)
                return action 
            else :
                action = np.argmax(self.values[obs])
                # ic(action)
                return action 
        else: 
            pass
     
     


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


    def learn(self, action_next,done):
        predict=self.last_reward+self.discount*(self.values[self.last_dest][action_next] )
        target=self.values[self.last_source][self.last_action]
        # ic(self.values[self.last_source])
        self.values[self.last_source][self.last_action]=self.values[self.last_source][self.last_action]+self.alpha*(predict-target)
        # ic(self.last_source)
        # ic(self.last_action)
        # ic(self.values[self.last_source])
        
        return  self.values

class DynaQ(object):
    def __init__(self, env, opt):
        self.opt=opt
        self.action_space = env.action_space
        self.env=env
        self.discount=opt.gamma
        self.alpha=opt.learningRate
        self.epsilon=0.1
        self.explo=opt.explo
        self.exploMode=opt.exploMode #0: epsilon greedy, 1: ucb
        self.sarsa=opt.sarsa
        self.modelSamples=opt.nbModelSamples
        self.test=False
        self.qstates = {}  # dictionnaire d'états rencontrés
        self.values = []   # contient, pour chaque numéro d'état, les qvaleurs des self.action_space.n actions possibles
        self.model = {} # model is a dictionary of dictionaries, which maps states to actions to 
                        # (reward, next_state) tuples
        self.planning_steps = 10
        




     

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
        action=0
        if self.exploMode==0:
            """epsilon greedy mode"""
            proba=random.random()
            if random.uniform(0,1) < self.explo : 
                action = env.action_space.sample()
                # ic(action)
                return action 
            else :
                action = np.argmax(self.values[obs])
                # ic(action)
                return action 
        else: 
            pass
     


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
        predict=self.last_reward+self.discount*np.max(self.values[self.last_dest][action])
        target=self.values[self.last_source][self.last_action]
        # ic(self.values[self.last_source])
        self.values[self.last_source][self.last_action]=self.values[self.last_source][self.last_action]+self.alpha*(predict-target)
        # ic(self.last_source)
        # ic(self.last_action)
        # ic(self.values[self.last_source])
        return  self.values

    def update_model(self,done):
        if not self.last_source in self.model:
            self.model[self.last_source] = {}
        if not self.last_action in self.model[self.last_source]:
            self.model[self.last_source][self.last_action] = {}
    
        self.model[self.last_source][self.last_action] = (self.last_dest,self.last_reward)
    
        
    def Planning_Steps(self):
        
            # ic(random.choice(list(self.model.keys())))
            

            s = random.choice(list(self.model.keys()))
            # ic((list(self.model[s].keys())))
            a = random.choice(list(self.model[s].keys()))
            (next_s,r) = self.model[s][a]
            predict = self.last_reward + self.discount*np.max(self.values[next_s])
            target=self.values[s][a]
            self.values[s][a] += self.alpha*(predict-target)




if args.Agent_Mode=='QLearning':

    if __name__ == '__main__':
        env,config,outdir,logger=init('./configs/config_qlearning_gridworld.yaml',"QLearning")

        freqTest = config["freqTest"]
        freqSave = config["freqSave"]
        nbTest = config["nbTest"]
        env.seed(config["seed"])
        np.random.seed(config["seed"])
        episode_count = config["nbEpisodes"]


        agent = QLearning(env, config)

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
                agent.learn(done)
                rsum += reward
                if done:
                    print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                    logger.direct_write("reward", rsum, i)
                    mean += rsum
                    break

      
        env.close()

if args.Agent_Mode=='Sarsa':

    if __name__ == '__main__':
        env,config,outdir,logger=init('./configs/config_qlearning_gridworld.yaml',"QLearning")

        freqTest = config["freqTest"]
        freqSave = config["freqSave"]
        nbTest = config["nbTest" ]
        env.seed(config["seed"])
        np.random.seed(config["seed"])
        episode_count = config["nbEpisodes"]

        agent=Sarsa(env, config)



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
                action = agent.act(ob) #agent.act(ob) rend l'action à faire. 

                new_ob, reward, done, _ = env.step(action)
                new_ob = agent.storeState(new_ob)
                action_next = agent.act(new_ob)
                # ic(action_next)

                j+=1

                if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                    done = True
                    #print("forced done!")

                agent.store(ob, action, new_ob, reward, done, j)

                agent.learn(action_next,done)
                rsum += reward
                
                if done:
                    # ic(j)
                    print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                    logger.direct_write("reward", rsum, i)
                    mean += rsum
                    break



        env.close()

if args.Agent_Mode=='DynaQ':
    if __name__ == '__main__':
        env,config,outdir,logger=init('./configs/config_qlearning_gridworld.yaml',"QLearning")

        freqTest = config["freqTest"]
        freqSave = config["freqSave"]
        nbTest = config["nbTest"]
        env.seed(config["seed"])
        np.random.seed(config["seed"])
        episode_count = config["nbEpisodes"]


        agent = DynaQ(env, config)

        rsum = 0
        mean = 0
        verbose = True
        itest = 0
        reward = 0
        planning_steps = 10
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

                agent.learn(done)
                agent.update_model(done)
                for _ in range(planning_steps):
                    agent.Planning_Steps()


                rsum += reward
             
                    


                if done:
                    print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                    logger.direct_write("reward", rsum, i)
                    mean += rsum
                    break



        env.close()