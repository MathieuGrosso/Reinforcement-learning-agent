
import matplotlib.pyplot as plt
import numpy as np 
import os 
import io
from icecream import ic
import random
import math



def read_file(filename):
    with open(filename, 'r') as f:
        ctx = []
        rwd = []
        for l in f.readlines():
            id, c, r = l.split(":")
            ctx.append([float(x) for x in c.split(";")])
            rwd.append([float(x) for x in r.split(";")])
    return ctx, rwd

filename='CTR.txt'
ctx,rwd= read_file(filename)

ctx = np.array(ctx)
rwd = np.array(rwd)
nb_articles, arms = rwd.shape
d = ctx.shape[1]



class Agent:
    def __init__(self, nb_bras: int, dim:int):
        self.nb_bras = nb_bras
        self.dim = dim
        self.t = []
        self.t_optimale =[]
        self.r = []
        self.time_step = 0
        self.time_step_optimale = 0

    def act(self, ctx):
        arm = random.randint(0, self.nb_bras-1)
        self.t.append(arm)
        self.time_step += 1
        return arm

    def ActOptimale(self,rwd,i):
        arm = np.where(rwd[i]==np.max(rwd[i]))
        self.t_optimale.append(arm)
        self.time_step_optimale+=1
        return arm 

    def update(self, arm, rwd):
        reward = rwd[self.time_step-1,arm]
        self.r.append(reward)
        return reward
    
    def update_optimale(self, arm, rwd):
        reward = rwd[self.time_step_optimale-1,arm]
        self.r.append(reward)
        return reward
    
class UCB: 
    def __init__(self, nb_bras: int, dim:int):
        self.nb_bras = nb_bras
        self.dim = dim
        self.t = [0 for arm in range(self.nb_bras)]
        self.t_optimale =[]
        self.r = []
        self.time_step = 0
        self.time_step_optimale = 0
        
        
    def act(self,rwd,c):
        # for arm in range(self.nb_bras):
        #     if self.t[arm] == 0:
        #         return arm

        self.time_step +=1
        ucb_values=[0.0 for arm in range(self.nb_bras)]
        for arm in range(self.nb_bras):
            UCB = np.sqrt((2*np.log(self.time_step)/float(self.t[arm])))
            ucb_values[arm]=rwd[c][arm]+UCB

        value_max = max(ucb_values)
        self.t[ucb_values.index(value_max)]+=1
  
        return ucb_values.index(value_max)

    def update(self,arm,rwd):
        reward = rwd[self.time_step-1,arm]
        self.r.append(reward)
        return reward
   
class EpsilonGreedy: 
    def __init__(self, nb_bras: int, dim:int,epsilon=0.05):
        self.nb_bras = nb_bras
        self.dim = dim
        self.t = []
        self.t_optimale =[]
        self.r = []
        self.time_step = 0
        self.time_step_optimale = 0
        self.epsilon=epsilon
    def act(self,rwd,c):
        if random.random() > self.epsilon : 
            return np.where(rwd[c]==(max(rwd[c])))
        else : 
            return int(random.random()*len(rwd[c]))
    
    def update(self,arm,rwd):
        reward = rwd[self.time_step-1,arm]
        self.r.append(reward)
        return reward

class linUCB: 
    def __init__(self, nb_bras: int, dim:int):
        self.nb_bras = nb_bras
        self.dim = dim
        self.t = [0 for arm in range(self.nb_bras)]
        self.t_optimale =[]
        self.r = []
        self.time_step = 0
        self.time_step_optimale = 0
        self.alpha=2
        
    def act(self,rwd,ctx,article):
   
        self.A = np.zeros((d, d*arms))
        self.b = np.zeros((d,arms))
        p = np.zeros((arms))
        first = []
        R = 0
        for arm in range(arms):
            self.x = np.reshape(ctx[article,:], (-1,1))
            if arm not in first:
                self.A[:d,d*arm:d*arm+d] = np.eye(d)
                first.append(arm)
            invA = np.linalg.inv(self.A[:d,d*arm:d*arm+d])
            theta = np.dot(invA,self.b[:d,arm])
            p[arm] = np.dot(theta.T,self.x)[0] + self.alpha*math.sqrt(np.dot(np.dot(self.x.T,invA),self.x)[0][0])
        arm = np.where(p==max(p))[0][0]
        # ic(arm)
        
        return arm
           

    def update(self,arm,rwd):
        # ic(self.time_step)
        R=0
        R += rwd[article,arm]
        self.A[:d,d*arm:d*arm+d] += np.dot(self.x,self.x.T)
        self.b[:d,arm] += np.reshape(np.dot(rwd[article,arm],self.x), self.b[:d,arm].shape)
        
        # self.r.append(reward)
        return R


#simulation for epsilon greedy:

greedy=0
numero_pub_greedy=[]
greedy_agent = EpsilonGreedy(10,5)
greedy_list = []
for c in range(nb_articles): 
    numero_pub_greedy.append(c)
    idx = greedy_agent.act(rwd,c)
    greedy += greedy_agent.update(idx,rwd)
    greedy_list.append(greedy)
ic(greedy)

            
        

# simulation for random: 

R = 0
random_list=[]
numero_pub_random=[]
agent = Agent(10,5)
for c in range(nb_articles):
    idx = agent.act(ctx)
    numero_pub_random.append(c)
    random_list.append(R)
    R += agent.update(idx, rwd)
    
ic(R)



# simulation for optimale: 

Optimale = 0
optimale_list=[]
numero_pub_optimale=[]
for c in range(nb_articles):
    numero_pub_optimale.append(c)
    idx = agent.ActOptimale(rwd,c)
    optimale_list.append(Optimale)
    Optimale += agent.update_optimale(idx, rwd)
    
ic(Optimale)


# simulation for UCB : 

ucb=0
ucb_agent = UCB(10,5)
ucb_list=[]
numero_pub_ucb=[]
for c in range(nb_articles): 
    idx = ucb_agent.act(rwd,c)
    numero_pub_ucb.append(c)
    ucb += ucb_agent.update(idx,rwd)
    ucb_list.append(ucb)
ic(ucb_list)
    


#simulation for linUCB # not working ? 

linucb=0
numero_pub=[]
linucb_list=[]
linucb_agent = linUCB(10,5)
for article in range(nb_articles): 
    numero_pub.append(c)
    idx = linucb_agent.act(rwd,ctx,article)
    linucb += linucb_agent.update(idx,rwd)
ic(linucb)






#plot results: 
plt.figure()

plt.subplot(221)
plt.plot(random_list,numero_pub_random)
plt.title("plot du  Random reward cumulé en fonction des articles")
plt.legend(["nombre article %s" % c, "Reward %s" % R], loc="upper left")


# plt.subplot(222)
# plt.plot(greedy_list,numero_pub_greedy)
# plt.title("plot du Greedy reward cumulé en fonction des articles")
# plt.legend(["nombre article %s" % numero_pub_greedy, "Reward %s" % greedy_list], loc="upper left")


plt.subplot(223)
plt.plot(optimale_list,numero_pub_optimale)
plt.title("plot du  OPTIMAL reward cumulé en fonction des articles")
plt.legend(["nombre article %s" % c, "Reward %s" % Optimale], loc="upper left")


plt.subplot(224)
plt.plot(ucb_list,numero_pub_optimale)
plt.title("plot du UCB reward cumulé en fonction des articles")
plt.legend(["nombre article %s" % c, "Reward %s" % ucb], loc="upper left")
plt.show()
