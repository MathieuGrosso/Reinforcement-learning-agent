import matplotlib
import pytorch
matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from icecream import ic 
import argparse

#parser
parser = argparse.ArgumentParser(prog='Value and Policy Iteration')
parser.add_argument("--Agent_Mode",help="allow to choose the mode of the agent", action="store")
parser.add_argument("--Plan",help="allow to chose the plan",action="store")
args=parser.parse_args()

print(args.Plan)

env = gym.make("gridworld-v0")
if args.Plan=="0":
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
if args.Plan=="1":
    env.setPlan("gridworldPlans/plan1.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
if args.Plan=="2":
    env.setPlan("gridworldPlans/plan2.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
if args.Plan=="3":
    env.setPlan("gridworldPlans/plan3.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
if args.Plan=="4":
    env.setPlan("gridworldPlans/plan4.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
if args.Plan=="5":
    ic("plan 5")
    env.setPlan("gridworldPlans/plan5.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
if args.Plan=="6":
    env.setPlan("gridworldPlans/plan6.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
if args.Plan=="7":
    env.setPlan("gridworldPlans/plan7.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
if args.Plan=="8":
    env.setPlan("gridworldPlans/plan8.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
if args.Plan=="9":
    env.setPlan("gridworldPlans/plan9.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
if args.Plan=="10":
    env.setPlan("gridworldPlans/plan10.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

env.seed(0)  #Initialise le seed du pseudo-random
env.render()  # permet de visualiser la grille du jeu
env.render(mode="human") #visualisation sur la console
states, mdp = env.getMDP()  # recupere le mdp et la liste d'etats
state, transitions = list(mdp.items())[0]
ic(state)



class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()


if args.Agent_Mode=='Random': 

    agent = RandomAgent(env.action_space)
    episode_count = 1000
    reward = 0
    done = False
    rsum = 0

    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            # state=env.getMDP()[0]
            # ic(state)
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()


def P(s0,a,s1):
    return mdp[s0][a][s1][0]
    
def R(s0,a,s1):
    return mdp[s0][a][s1][2]
      
  
class ValueIteration: 
    def __init__(self,action_space,states,mdp):
        # env.reset()
        # self.V0=0
        self.action_space = action_space
        self.gamma=0.999
        self.policy=0
        self.epsilon=0.05
        self.states, self.mdp = env.getMDP()
        self.n_next_states = len(self.mdp[0][0])
        self.n_actions = len(list(mdp.items())[0][1].keys())
        # ic(self.n_actions)
        self.V=[0.00 for state in range(len(self.states))] 
        self.Vi_1=[-1 for state in range(len(self.states))]

 
    def compute_value(self):
        i=0
        # V=[0.06 for state in range(len(self.states))] # on initialise V qui vaut 0 pour chaque état au début. 
        # Vi_1=[0.0 for state in range(len(self.states))]
        # ic(V)
        # ic(Vi_1)
        ic(np.linalg.norm(np.array(self.V)-np.array(self.Vi_1)))
        while np.linalg.norm(np.array(self.V)-np.array(self.Vi_1)) > 0.05 :
            ic(np.linalg.norm(np.array(self.V)-np.array(self.Vi_1)))
            if i !=0 :
                self.Vi_1 = self.V.copy()
            i+=1
            for current_state in self.mdp.keys():
                # ic(current_state)
                actualisation=[0 for i in range(self.n_actions)]
                for action in range(self.n_actions):
                    # ic(self.n_actions)
                    # ic(action)
                    next_states = []
                    for j in range(self.n_next_states):
                        next_states.append(self.mdp[current_state][action][j][1])
                        # ic(next_states)
                        
                    for next_state in range(len(next_states)):
                            ic(self.Vi_1[next_states[next_state]])
                            actualisation[action] += P(current_state,action,next_state)*(R(current_state,action,next_state)+self.gamma*self.Vi_1[next_states[next_state]])
                            # ic(P(current_state,action,next_state))
                            # ic(R(current_state,action,next_state))
                            ic([next_states[next_state]])
                            # ic(actualisation)
                            
                self.V[current_state] = max(actualisation)
                # ic(self.V)
                # ic(self.Vi_1)
        print("time to converge:",i)
        return self.V
            
    def find_policy(self,V): 
        policy=[0.0 for state in range(len(self.states))] 
        for current_state in self.mdp.keys():
            actualisation=[0 for i in range(self.n_actions)]
            for action in range(self.n_actions):
                next_states = []
                for j in range(self.n_next_states):
                    next_states.append(self.mdp[current_state][action][j][1])
                for next_state in range(len(next_states)):
                    # ic(next_states)
                    actualisation[action] += P(current_state,action,next_state)*(R(current_state,action,next_state)+self.gamma*V[next_states[next_state]])
                        # actualisation[action] = [sum(find_proba_reward_transition(self.mdp,current_state, action, next_state)+self.gamma*self.Vi_1[next_state] for next_state in mdp.keys())]
                    # ic(actualisation)
            policy[current_state] = np.argmax(actualisation) 
        # ic(policy)
        return policy
    
    def act(self, observation):
        return self.action_space.sample()


if args.Agent_Mode=='ValueIteration' :
    
    # Execution avec un Agent

    Agent = ValueIteration(env.action_space,states,mdp)
    V=Agent.compute_value()
    ic(Agent.find_policy(V))

    episode_count = 1000
    reward = 0
    done = False
    rsum = 0

    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = Agent.act(obs)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            # state=env.getMDP()[0]
            # ic(state)
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()


class PolicyIteration: 
    def __init__(self,action_space,states,mdp):
        self.action_space = action_space
        self.gamma=0.9
        self.policy=0
        self.epsilon=0.05
        self.states, self.mdp = env.getMDP()
        self.n_next_states = len(self.mdp[0][0])
        self.n_actions = len(list(mdp.items())[0][1].keys())
        # ic(self.n_actions)
        # self.V=[0.00 for state in range(len(self.states))] 
        # self.v=[-1 for state in range(len(self.states))]
        self.POLICY=[0 for state in range(len(self.states))] 
        # self.POLICY=np.random.random_integers(low=0,high=3,size=(len(self.states)))
        # self.POLICY[1]=0
        # self.POLICY[4]=0
        # ic(self.POLICY)
        self.policy=[-1 for state in range(len(self.states))] 

    def compute_value(self):
        i=0
        self.V=[0.00 for state in range(len(self.states))] 
        self.v=[-1 for state in range(len(self.states))]
        # ic(np.linalg.norm(np.array(self.V)-np.array(self.v)) > self.epsilon)
        while np.linalg.norm(np.array(self.V)-np.array(self.v)) > self.epsilon :
            # ic(self.V)
            # ic(np.linalg.norm(np.array(self.V)-np.array(self.v)) > self.epsilon)
            # ic(self.v)
            if i !=0 :
                self.v = self.V.copy()
            i+=1
            # ic(i)
            for current_state in self.mdp.keys():
                actualisation=0
                action=self.POLICY[current_state]
                next_states=[]
                for j in range(self.n_next_states):
                    next_states.append(self.mdp[current_state][action][j][1])
                    # ic(next_states)
                
                for next_state in range(len(next_states)):
                    # ic(range(len(next_states)))
                    # ic(next_states[next_state])
        
                    actualisation    += P(current_state,action,next_state)*(R(current_state,action,next_state)+self.gamma*self.v[next_states[next_state]])
                    # ic(actualisation)
                self.V[current_state] = actualisation
            
                # ic(self.V)
                # ic(self.v)
        print("time to converge:",i)
        return self.V
            
    def find_policy(self,V):  
        policy=[0.0 for state in range(len(self.states))] 
        for current_state in self.mdp.keys():
            actualisation=[0 for i in range(self.n_actions)]
            for action in range(self.n_actions):
                next_states = []
                for j in range(self.n_next_states):
                    next_states.append(self.mdp[current_state][action][j][1])
                for next_state in range(len(next_states)):
                    actualisation[action] += P(current_state,action,next_state)*(R(current_state,action,next_state)+self.gamma*V[next_states[next_state]])
            policy[current_state] = np.argmax(actualisation) 
        return policy


    def final_policy(self):
        k=0
        while np.linalg.norm(np.array(self.POLICY)-np.array(self.policy)) !=0 :
        
            if k != 0: 
                self.policy=self.POLICY.copy()
            k+=1
            V=self.compute_value()
            self.POLICY=self.find_policy(V)
            # ic(self.POLICY)
            # ic(self.policy)
        print("time for the algorithm to converge:",k)
        return self.POLICY

    def act(self, observation):
        return self.action_space.sample()

if args.Agent_Mode=='PolicyIteration' :

    
    # Execution avec un Agent
    Agent=PolicyIteration(env.action_space,states,mdp)
    policy2=Agent.final_policy()
    ic(policy2)


    episode_count = 1000
    reward = 0
    done = False
    rsum = 0

    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = Agent.act(obs)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            # state=env.getMDP()[0]
            # ic(state)
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()

