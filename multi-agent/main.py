import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
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
from datetime import datetime
from MultiAgentDDPG import Agent, MADDPG
from collections import deque


"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    world.dim_c = 0
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    print("env ok!")
    return env #,scenario,world

def padObs(obs,size):
    return([np.concatenate((o,np.zeros(size-o.shape[0]))) if o.shape[0]<size else o for o in obs])

if __name__ == '__main__':

    env, config, outdir, logger = init('./configs/config_maddpg_simple_spread.yaml', "MADDPG")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]


    noise = [config["noise"](config["action_size"], sigma=config["sigma_noise"]) for i in range(env.n)]
    noiseTest = [config["noise"](config["action_size"], sigma=config["sigma_noiseTest"]) for i in range(env.n)]
    #
    #
    obs = env.reset()
    obs_n = [o.shape[0] for o in obs]
    mshape=max(obs_n)
    obs_n = [mshape for o in obs]
    dim_a = [2] * env.n  # force applied on x, y axis (in [-1, 1])
    first_obs = env.reset()
    dim_s = [len(obs_i) for obs_i in first_obs]
    agent = MADDPG(env, config, config["action_size"], obs_n, noise, noiseTest,  hidden_sizes_q=list(config.hidden_sizes_q),
                     hidden_sizes_mu=list(config.hidden_sizes_mu),dimIn = dim_s,dim_a = dim_a,dimOut=dim_a)

    verbose=True
    rsum = np.zeros(env.n)
    mean = np.zeros(env.n)
    itest = 0
    ntest = 0
    ntrain = 0
    rewards = np.zeros(env.n)
    done = [False]
    ended=False
    agent.nbEvents = 0
    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = np.zeros(env.n)

        obs = env.reset()
        obs = padObs(obs, mshape)
        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = np.zeros(env.n)
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            #logger.direct_write("rewardTest", mean / nbTest, itest)
            for k in range(env.n):
                logger.direct_write("rewardTest/raw_" + str(k), mean[k]/nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()


        new_obs = np.array(obs)

        ended = False

        while True:

            if verbose:
                env.render(mode="none")

            obs = new_obs
            actions = agent.act(obs)

            j += 1
            new_obs, rewards, done, _ = env.step(actions)
            new_obs = padObs(new_obs, mshape)
            if ((not agent.test) and j >= int(config["maxLengthTrain"])) or (j>=int(config["maxLengthTest"])) :
                ended=True
            new_obs = np.array(new_obs)
            rewards = np.array(rewards)
            if (not agent.test):

                agent.store(obs, actions, new_obs, rewards, done,j)

                if agent.timeToLearn(ended):
                    agent.learn()

            rsum += rewards

            if done[0] or ended:
                agent.endsEpisode()
                if not agent.test:
                    ntrain += 1
                    print("Train:",str(ntrain) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                    for k in range(len(rsum)):
                        logger.direct_write("reward/raw_"+str(k), rsum[k], ntrain)
                else:
                    ntest += 1
                    print("Test:", str(ntest) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")


                mean = mean + rsum
                rsum = np.zeros(env.n)

                break


    env.close()

