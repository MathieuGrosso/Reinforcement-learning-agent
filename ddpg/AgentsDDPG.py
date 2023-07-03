import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
#matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from torch import optim
from noise import * 
from memory import *



class Agent:
    """The world's simplest agent!"""

    def __init__(self, env, opt):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents=0


    def act(self, obs):
        a=self.action_space.sample()
        return a

    # sauvegarde du modèle
    def save(self,outputDir):
        pass

    # chargement du modèle.
    def load(self,inputDir):
        pass

    # apprentissage de l'agent. Dans cette version rien à faire
    def learn(self):
        # sampler les mini batch et faire la descente de gradient ici.
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            pass
        pass

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = (ob, action, reward, new_ob, done)
            self.lastTransition=tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        if done:
            return True
        else:
            return False


    




class DDPG(Agent):
    def __init__(self,env,opt,dimIn,dim_a,dimOut,logger):
        super(DDPG,self).__init__(env,opt)
        
        self.batch_size = opt.batch_size
        self.gamma = opt.gamma
        self.tau = opt.tau
        self.eps_std = opt.eps_std
        self.alow = opt.alow
        self.ahigh = opt.ahigh
        self.buffer_limit = opt.buffer_limit
        self.action = opt.action 
        self.Critic = Critic(inSize=dimIn,dim_a=dim_a)
        self.Critic_target = Critic(inSize=dimIn,dim_a=dim_a)
        self.Actor     = Actor(inSize = dimIn,dim_a=dimOut)
        self.Actor_target     = Actor(inSize = dimIn,dim_a=dimOut)
        self.optimizer_actor  = optim.Adam(self.Actor.parameters(), lr=opt.lr_actor)
        self.optimizer_critic = optim.Adam(self.Critic.parameters(), lr=opt.lr_critic)
        self.memory    = ReplayBuffer(self.buffer_limit)
        self.criterion = torch.nn.HuberLoss()
        self.criterion2 = F.smooth_l1_loss

        #initialize weights of the target networks with the current networks: 
        self.Actor_target.load_state_dict(self.Actor.state_dict())
        self.Critic_target.load_state_dict(self.Critic.state_dict())

    def act(self,obs):
        if self.action=='OU':
            ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
            a = self.Actor(self._t(obs)) 
            a = a + ou_noise()[0]
            # ic(a)
            a=self._t(a)
            return a
        if self.action == 'Gaussian':
            obs = self._t(obs)
            with torch.no_grad():
                probs = self.Actor(obs)
            epsilon = torch.normal(mean=0, std=self.eps_std, size=probs.shape)
            a=torch.clamp(probs+epsilon,min = self.alow,max = self.ahigh)
            return a

    


    def _soft_update(self,net,net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
    
    def store(self,ob, action, new_ob, reward, done):
        self.episode_count = episode_count
        if not self.test:
            tr=(ob, action, new_ob, reward, done)
            self.memory.put(transition=tr)

    def _t(self,x):
        return torch.tensor(x,dtype=torch.float)


    def learn(self):
        #randomly sample a batch of transitions B
        ob, action, reward, next_ob, done = self.memory.sample(self.batch_size)

        # compute target 
        target = reward + self.gamma * self.Critic_target(next_ob, self.Actor_target(next_ob)) * (1-done)
        critic_loss = self.criterion2(self.Critic(ob,action.unsqueeze(-1)),target.detach())
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        #update policy by ONE step of gradient ASCENT
        actor_loss = -self.Critic(ob,self.Actor(ob)).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        #update both target networks soft
        self._soft_update(self.Actor,self.Actor_target)
        self._soft_update(self.Critic,self.Critic_target)
        
        


    



if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_DDPG_pendulum.yaml', "DDPG")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    # dim_a, = env.action_space.shape
    dim_s, = env.observation_space.shape
    dim_a, = env.action_space.shape
    agent = DDPG(env,config,dim_s,dim_a,dim_a,logger)


    rsum = 0
    mean = 0
    verbose = True
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

        new_ob = agent.featureExtractor.getFeatures(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action= agent.act(ob)
            # ic(action)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j+=1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob, action,  reward,new_ob, done)
            rsum += reward

            if agent.memory.size() >= 2000 and agent.timeToLearn(done):
                print(j)
                agent.learn()
            

            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close() 
    