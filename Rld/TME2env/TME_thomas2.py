import matplotlib
matplotlib.use("TkAgg")
import gym
import numpy as np
import gridworld
from icecream import ic


env = gym.make("gridworld-v0")
env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
states, mdp = env.getMDP()  # recupere le mdp et la liste d'etats

for i in range(len(states)):
    if i not in list(mdp.keys()):
        mdp[i] = {0: [(0., 0, 0., False), (0., 0, 0., False), (0., 0, 0., False)], 1: [(0., 0, 0., False), (0., 0, 0., False), (0., 0, 0., False)], 2: [(0., 0, 0., False), (0., 0, 0., False), (0., 0, 0., False)], 3: [(0., 0, 0., False), (0., 0, 0., False), (0., 0, 0., False)]}

n_actions = len(mdp[0])
n_states = len(mdp.keys())
n_next_states = len(mdp[0][0])
ic(n_next_states)
ic(states)

def P(s0,a,s1):
    return mdp[s0][a][s1][0]

def R(s0,a,s1):
    return mdp[s0][a][s1][2]

class ValueIteration():
    def __init__(self):
        self.V_current = [0]*n_states
        self.V_next = [1]*n_states
        self.policy = [0]*n_states
        self.epsilon = 0.005
        self.gamma = 0.999
        

    def act(self, obs):
        return self.policy[states.index(gridworld.GridworldEnv.state2str(obs))]

    def computeValue(self):
        i = 0
        while np.linalg.norm(np.array(self.V_current)-np.array(self.V_next)) > self.epsilon:
            if i !=0 :
                self.V_current = self.V_next.copy()
            for current_state in range(n_states):
                Sum = [0]*n_actions
                for action in range(n_actions):
                    next_states = []
                    for j in range(n_next_states):
                        next_states.append(mdp[current_state][action][j][1])
                    for next_state in range(len(next_states)):
                        Sum[action] += P(current_state,action,next_state)*(R(current_state,action,next_state)+self.gamma*self.V_current[next_states[next_state]])
                self.V_next[current_state] = max(Sum)
            i += 1
        for current_state in range(n_states):
            Sum = [0]*n_actions
            for action in range(n_actions):
                next_states = []
                for j in range(n_next_states):
                    next_states.append(mdp[current_state][action][j][1])
                for next_state in range(len(next_states)):
                    Sum[action] += P(current_state,action,next_state)*(R(current_state,action,next_state)+self.gamma*self.V_current[next_states[next_state]])
            self.policy[current_state] = Sum.index(max(Sum))
        return self.policy

agent = ValueIteration()
ic(agent.computeValue())

episode_count = 1000
reward = 0
done = False
rsum = 0

# for i in range(episode_count):
#     obs = env.reset()
#     env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
#     if env.verbose:
#         env.render()
#     j = 0
#     rsum = 0
#     while True:
#         action = agent.act(obs) #actions possibles
#         obs, reward, done, _ = env.step(action)
#         rsum += reward
#         j += 1
#         if env.verbose:
#             env.render()
#         if done:
#             print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
#             break

# print("done")
env.close()