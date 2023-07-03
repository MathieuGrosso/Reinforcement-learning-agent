
import argparse
from os import X_OK, truncate
import sys
import matplotlib
#matplotlib.use("Qt5agg")
# matplotlib.use("TkAgg")
import gym
import gridworld
import torch
import random
import torch.nn as nn

from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
# import highway_env
from matplotlib import pyplot as plt
import yaml
from icecream import ic
from datetime import datetime

from ActorCritic import Batch_ActorCritic



if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_random_cartpole.yaml', "Batch_ActorCritic")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

   
    agent = Batch_ActorCritic(env,config,logger = logger, mode='TD0',test_mode=False,layer =range(30) )


    rsum = 0
    mean = 0
    verbose = True
    target_update=100
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



        while True: 
            if verbose:
                env.render()

            new_ob,reward,done = agent.act(ob)
            # ic(reward)
            j+=1

            ob = new_ob 
            agent.update(i)

     
            if agent.timeToLearn(done):
                

                agent.learn()
            if done: 
                agent.restart_transition
            
           
            rsum += reward

            
 
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close()
