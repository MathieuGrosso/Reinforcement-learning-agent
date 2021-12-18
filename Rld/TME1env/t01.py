import numpy as np
import math
import re
import random
import matplotlib.pyplot as plt

CTR = "./CTR.txt"

with open(CTR, "r") as f:
    content = f.readlines()
    
content = [x.split(':') for x in content]

articles = [x[1].split(';') for x in content]
clicks = [x[2].split(';') for x in content]

articles = [[float(x_i) for x_i in x] for x in articles]
clicks = [[float(x_i) for x_i in x] for x in clicks]

clicks = np.array(clicks)
articles = np.array(articles)


class Agent:
    def __init__(self, strat, articles, clicks, delta=0.05, alpha=None):
        self.strat = strat
        self.articles = articles
        self.clicks = clicks
        self.score = 0
        self.static_best = np.argmax(np.sum(clicks, axis=0))
        
        # for UCB
        self.reward_history = np.zeros(10)
        self.choice_history = np.zeros(10)
        
        # LinUCB
        self.A_list = [np.eye(5) for _ in range(10)]
        self.b_list = [np.zeros(5) for _ in range(10)]
        if alpha is None:
            self.alpha = 1 + np.sqrt(np.log(2 / delta) / 2)
        else:
            self.alpha = alpha
            
        # Thompson Sampling for Contextual Bandits (Agrawal & Goyal, 2013):
        # not applicable here, as we have a global context vector x_t all machines (announcers),
        # not individual context vectors x_it
        
        
    def act(self, t):
        if self.strat == 'Random':
            return random.randint(0,9)
        
        if self.strat == 'StaticBest':
            return self.static_best
        
        if self.strat == 'Optimal':
            return np.argmax(clicks[t])
        
        if self.strat == 'UCB':
            ub_estimate = [1/self.choice_history[i]*self.reward_history[i] 
                             + np.sqrt(2*np.log(t) / self.choice_history[i]) for i in range(10)]
            
            choice = np.argmax(ub_estimate)
            
            self.reward_history[choice] += self.clicks[t, choice]
            self.choice_history[choice] += 1
            return choice
        
        if 'LinUCB' in self.strat:
            x_t = self.articles[t]
            piche = np.zeros(10)
            
            for i in range(10):
                A = self.A_list[i]
                b = self.b_list[i]
                A_inv = np.linalg.inv(A)
                
                theta = A_inv.dot(b)
                piche[i] = theta.dot(x_t) + self.alpha * np.sqrt(x_t.dot(A_inv).dot(x_t))
                
            choice = np.argmax(piche)
            r_t = self.clicks[t, choice]

            self.A_list[choice] += x_t.dot(x_t)
            self.b_list[choice] += r_t * x_t
            return choice
        
      
                
        
    def reward(self, t):
        return clicks[t][self.act(t)]


optimize_ucb = True # if true: check all alpha values for lin ucb, if false: check all different models and alpha = 0.15 for lin ucb. 

if optimize_ucb:
    # Then we optimize lin UCB : 
    strat_names = ['UCB','optimale','LinUCB']
    strategies = {}
    alpha_values = np.logspace(-2, -0.5, num=10)
    for strat in strat_names:
        if 'LinUCB' in strat:
            for alpha in alpha_values:
                strategies[strat,alpha] = Agent(strat, articles, clicks, alpha=alpha)
        else:
            strategies[strat,0] = Agent(strat, articles, clicks)


    total_reward = []
    total_regret = []

    for t in range(len(clicks)):
        rewards = []

        
        for strat in strat_names:
            for alpha in alpha_values:
                i = strategies[strat,alpha].act(t)
                r_ti = clicks[t, i]
                rewards.append(r_ti)

            
        total_reward.append(rewards)

        
    total_regret = np.cumsum(np.array(total_regret), axis=0)
    total_reward = np.cumsum(np.array(total_reward), axis=0)


    plt.figure(figsize=(10, 10))

    for k, s in enumerate(alpha_values):
        plt.plot(total_reward[:, k], label=s)

    plt.legend()
    plt.show()

else : 

    #first we test the five models: 
    strat_names = ['Random', 'StaticBest', 'Optimal', 'UCB', 'LinUCB-0.15']
    strategies = {}
    for strat in strat_names:
        if 'LinUCB' in strat:
            alpha = float(strat.split('-')[1])
            strategies[strat,alpha] = Agent(strat, articles, clicks, alpha=alpha)
        else:
            strategies[strat] = Agent(strat, articles, clicks)

    total_reward = []


    for t in range(len(clicks)):
        rewards = []


        for strat in strat_names:
            i = strategies[strat].act(t)
            r_ti = clicks[t, i]
            rewards.append(r_ti)
        total_reward.append(rewards)

    total_reward = np.cumsum(np.array(total_reward), axis=0)


    plt.figure(figsize=(10, 10))

    for k, s in enumerate(strat_names):
        plt.plot(total_reward[:, k], label=s)

    plt.legend()
    plt.show()