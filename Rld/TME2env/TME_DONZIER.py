import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy


def init_env(plan_id=0, Rs=-0.01, verbose=False):
    env = gym.make("gridworld-v0")
    env.setPlan(f"gridworldPlans/plan{plan_id}.txt",
                {0: Rs, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random

    if verbose:
        print(env.action_space)  # Quelles sont les actions possibles
        print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
        env.render()  # permet de visualiser la grille du jeu 
        env.render(mode="human") #visualisation sur la console

    statedic, mdp = env.getMDP()  # recupere le mdp : statedic

    if verbose:
        print(f"Nombre d'etats : {len(statedic)}")  # nombre d'etats ,statedic : etat-> numero de l'etat
        state, transitions = list(mdp.items())[0]
        print(state)  # un etat du mdp
        print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    return env, mdp


env, mdp = init_env()

def play_one_game(agent, env, reward, done):
    obs = env.reset()
    j = 0
    rsum = 0
    while True:
        action = agent.act(obs, reward, done)
        obs, reward, done, _ = env.step(action)
        rsum += reward
        j += 1
        if done:
            return rsum, j

def play_games(agent, episode_count=5):
    reward = 0
    done = False
    rsum = 0
    for i in range(episode_count):
        rsum, j = play_one_game(agent, env, reward, done)
        print(f"Episode {i}: rsum= {rsum}, {j} actions")

    print("done")
    env.close()


dict_cells = {"case vide"       : 0,
              "mur"             : 1,
              "joueur"          : 2,
              "sortie"          : 3,
              "objet a ramasser": 4,
              "piege mortel"    : 5,
              "piege non mortel": 6}

def get_grid(d, bg=0., reward_int=0):
    state = list(d.keys())[0]
    state = gridworld.GridworldEnv.str2state(state)
    grid = np.full(state.shape, bg)

    for k, v in d.items():
        state = gridworld.GridworldEnv.str2state(k)
        if np.sum(state == dict_cells["objet a ramasser"]) != reward_int:
            continue
        x, y = np.where(state == dict_cells["joueur"])
        grid[x[0], y[0]] = v

    return grid



actions_dict = {0: 'South', 1: 'North', 2: 'West', 3: 'East'}
actions_grad = {0: [0, -1], 1: [0, 1], 2:[-1, 0], 3:[1, 0]}

def get_directions(policy):
    state = list(policy.keys())[0]
    state = gridworld.GridworldEnv.str2state(state)
    X = np.zeros(state.shape)
    Y = np.zeros(state.shape)

    player_id = 2
    for k, v in policy.items():
        state = gridworld.GridworldEnv.str2state(k)
        x, y = np.where(state == player_id)
        X[x[0], y[0]] = actions_grad[v][0]
        Y[x[0], y[0]] = actions_grad[v][1]

    X = 0.05 * X
    Y = 0.05 * Y

    return X, Y

def plot_policy(agent, reward_int=0):
    state = list(agent.policy.keys())[0]
    state = gridworld.GridworldEnv.str2state(state)
    shape = state.shape

    plt.title(f"Policy with gamma={agent.gamma}")

    X, Y = np.meshgrid([s for s in range(shape[0])], [s for s in range(shape[1])])

    evaluations = get_grid(agent.evaluations, reward_int=reward_int)
    plt.imshow(evaluations)

    grad = get_directions(agent.policy)
    plt.colorbar()
    plt.quiver(grad[0], grad[1], scale=0.7)
    plt.show()

def plot_map(plan_id):
    img = env._gridmap_to_img(env.reset())

    plt.title(f"Map {plan_id}")
    plt.imshow(img)
    plt.show()


from scipy.spatial.distance import euclidean


def distance_v_valeurs(v1, v2):
    """
    Retourne la distance entre deux
    dictionnaires associés à des V-valeurs.
    """
    return euclidean(np.array(list(v1.values())),
                     np.array(list(v2.values())))


class ValueIteration(object):
    """
    Agent implementing the Value iteration algorithm
    to determine the policy.
    """

    def __init__(self, mdp, gamma, eps):
        self.mdp = mdp
        self.gamma = gamma
        self.eps = eps
        self.policy = self.compute_policy()

    def _compute_Bellman(self, V, s, a):
        return sum(v[0] * (v[2] + self.gamma * (V[v[1]] if not v[3] else 0))
                   for v in self.mdp[s][a])

    def _compute_v(self):
        states = mdp.keys()
        v_k      = {s: 0 for s in states}
        v_k_next = {}

        iterate = True
        while iterate:
            v_k_next = {s: max([self._compute_Bellman(v_k, s, a)
                             for a in self.mdp[s].keys()])
                        for s in states}
                        
            iterate = distance_v_valeurs(v_k, v_k_next) > self.eps
            v_k = v_k_next.copy()

        return v_k_next

    def compute_policy(self):
        states = mdp.keys()
        policy = {}
        self.evaluations = {}

        V = self._compute_v()
        for s in self.mdp.keys():
            vs = {a: self._compute_Bellman(V, s, a) for a in self.mdp[s].keys()}
            policy[s] = max(vs.items(), key=lambda x:x[1])[0]
            self.evaluations[s] = max(vs.items(), key=lambda x:x[1])[1]

        return policy

    def act(self, observation, reward, done):
        return self.policy[gridworld.GridworldEnv.state2str(observation)]

gamma = 0.999
eps = 0.01
agent = ValueIteration(mdp, gamma, eps)
play_games(agent)
plot_policy(agent)