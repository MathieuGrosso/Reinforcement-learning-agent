import datetime
import glob
import pickle
import shutil
from pathlib import Path
import numpy as np
import gym
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from nets import Pi



class ExpertAgent:
    def __init__(self, dim_a, dim_s, file):
        self.dim_a = dim_a
        self.dim_s = dim_s
        with open(file, 'rb') as handle:
            self.expert_data = pickle.load(handle)  # FloatTensor
            self.states = self.expert_data[:, : self.dim_s]
            self.actions = self.expert_data[:, self.dim_s:]
            self.states = self.states.contiguous()
            self.actions = self.actions.contiguous()
            self.actions = self.actions.argmax(dim=1)  # index instead of one-hot

#define Behavioral Cloning agent : 
class BC:
    def __init__(self, dim_s, dim_a, lr):
        self.pi = Pi(dim_s,dim_a)
        self.optim = torch.optim.Adam(self.pi.parameters(), lr=lr)

    def act(self, obs):
        logits = self.pi(torch.tensor(obs, dtype=torch.float32))
        return torch.argmax(logits).item()

    def learn(self, states, actions):
        policy = Categorical(logits=self.pi(states))

        # simply maximize log likelihood of selection the expert's actions
        log_prob = policy.log_prob(actions)
        loss = - log_prob.sum()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()


class GAIL : 
    def __init__(self,dim_s,dim_a,lr,K,clip_eps,entropy_weight):
        self.dim_a = dim_a
        self.dim_s = dim_s
        self.epoch = K 
        self.clip_eps = clip_eps
        self.entropy_weight = entropy_weight # terme d'entropy rajouter pour eviter que l'on converge trop vite
        self.D = Pi(dim_s = dim_s + dim_a,dim_a = 1)
        self.pi = Pi(dim_s = dim_s,dim_a = dim_a)
        self.V = Pi(dim_s,1)
        self.criterion = nn.BCELoss()

        self.optimizerD = torch.optim.Adam(self.D.parameters(), lr=lr)
        self.optimizerP = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.optimizerV = torch.optim.Adam(self.V.parameters(), lr=lr)


    def _t(self,x):
        return torch.tensor(x,dtype=torch.float)

    def act(self, obs):
        logits = self.pi(torch.from_numpy(obs).float())
        multinomial = Categorical(logits=logits)
        return multinomial.sample().item()

    # def learn(self, batch_agent, batch_expert):
    #     obs, act, r_cumulated, obs_next, _ = batch_agent
    #     act_one_hot = F.one_hot(act.flatten(), num_classes=self.dim_a)
    #     obs_expert, act_expert = batch_expert
    #     act_expert_one_hot = F.one_hot(act_expert, num_classes=self.dim_a)

    #     # --- Discriminator step
    #     input_expert = torch.cat([obs_expert, act_expert_one_hot], dim=1)
    #     input_agent = torch.cat([obs, act_one_hot], dim=1)
    #     noise_expert = torch.normal(0, 0.01, size=input_expert.shape)
    #     noise_agent = torch.normal(0, 0.01, size=input_agent.shape)
    #     d_expert = torch.sigmoid(self.D(input_expert + noise_expert))
    #     d_agent = torch.sigmoid(self.D(input_agent + noise_agent))
    #     loss_d = (F.binary_cross_entropy_with_logits(d_expert, torch.ones_like(d_expert)) +
    #               F.binary_cross_entropy_with_logits(d_agent, torch.zeros_like(d_agent)))
    #     assert not loss_d.isnan()
    #     self.optimizerD.zero_grad()
    #     loss_d.backward()
    #     self.optimizerD.step()

    #     # --- Policy step (with KL clipped PPO)
    #     # probas of the policy used to obtain the samples
    #     pi_k_logits = self.pi(obs).detach()
    #     pi_k = F.softmax(pi_k_logits, dim=1)
    #     pi_k_a = pi_k.gather(1, act)
    #     loss_pi_list = []
    #     loss_v_list = []
    #     entropy_list = []

    #     # rewards_clipped = torch.clamp(torch.log(d_agent), min=-100, max=0)
    #     # td_target = (rewards_clipped + 0.99 * self.v(obs_next) * not_finished).detach()
    #     # advantage = (td_target - self.v(obs)).detach()
    #     advantage = (r_cumulated - self.V(obs)).detach()
    #     for s in range(self.epoch):
    #         # train pi
    #         pi_logits = self.pi(obs)
    #         pi = F.softmax(pi_logits, dim=1)
    #         pi_a = pi.gather(1, act)

    #         ratio = pi_a / pi_k_a
    #         loss_1 = (ratio * advantage)
    #         loss_2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
    #         entropy = Categorical(pi).entropy()  # entropy term that we wish to maximize to push for exploration
    #         loss_pi = (-torch.min(loss_1, loss_2) - self.entropy_weight * entropy).mean()
    #         assert not loss_pi.isnan()
    #         self.optimizerP.zero_grad()
    #         loss_pi.backward()
    #         self.optimizerP.step()

    #         # train v_pi
    #         loss_v = F.smooth_l1_loss(self.V(obs), r_cumulated)
    #         self.optimizerV.zero_grad()
    #         loss_v.backward()
    #         self.optimizerV.step()

    #         loss_pi_list.append(loss_pi.item())
    #         loss_v_list.append(loss_v.item())
    #         entropy_list.append(entropy.mean().item())

    #     return {'loss_discriminator': loss_d.item(),
    #             'loss_actor': np.mean(loss_pi_list),
    #             'loss_critic': np.mean(loss_v_list),
    #             'd_expert': d_expert.mean().item(),
    #             'd_agent': d_agent.mean().item(),
    #             'entropy': np.mean(entropy_list)}

    def learn(self,batch_agent,batch_expert):
        # sample data : 
        s,a,r_cum,s_prime,_ = batch_agent 
        a_one_hot = F.one_hot(a.flatten(), num_classes=self.dim_a)
        s_expert,a_expert = batch_expert
        a_expert_one_hot = F.one_hot(a_expert,num_classes = self.dim_a)

        # Update Discriminator : 
        input_expert = torch.cat([s_expert,a_expert_one_hot],dim=1)
        input_agent = torch.cat([s,a_one_hot],dim = 1)
        ## adding noise : 
        noise_expert = torch.normal(0, 0.01, size=input_expert.shape)
        noise_agent = torch.normal(0, 0.01, size=input_agent.shape)
        d_expert = torch.sigmoid(self.D(input_expert + noise_expert))
        d_agent = torch.sigmoid(self.D(input_agent + noise_agent))

        true = torch.ones_like(d_expert)
        fake = torch.zeros_like(d_agent) 

        loss_d = (self.criterion(d_expert, true) +
                  self.criterion(d_agent, fake))
                  
        assert not loss_d.isnan()

        self.optimizerD.zero_grad()
        loss_d.backward()
        self.optimizerD.step()

        ## policy Step using PPO clipped/  PPO KL 


        # probas of the policy used to obtain the samples
        pi_k_logits = self.pi(s).detach()
        pi_k = F.softmax(pi_k_logits, dim=1)
        pi_k_a = pi_k.gather(1, a)
        loss_pi_list = []
        loss_v_list = []
        entropy_list = []
        advantage = (r_cum - self.V(s)).detach()

        for _ in range(self.epoch):
            # train pi
            pi_logits = self.pi(s)
            pi = F.softmax(pi_logits, dim=1)
            entropy = Categorical(pi).entropy() # on veut maximiser ce terme d'entropie pour eviter de converger vers des solutions sous optimales 
            pi_a = pi.gather(1, a)


            prob_ratio = torch.exp(pi_a - pi_k_a).flatten() #exp de log prob pour faire la division. 
            weighted_probs = advantage * prob_ratio
            clamped_prob_ratio = torch.clamp(prob_ratio,1-self.clip_eps,1+self.clip_eps)
            clamped_weighted_ratio = advantage * clamped_prob_ratio
            
            regularization =  - (self.entropy_weight * entropy).mean() # on maximise ces deux terms pour pusher l'exploration et augmenter le reward. 
            loss_1 = -torch.minimum(weighted_probs,clamped_weighted_ratio).mean()
            loss_pi = loss_1 +regularization
            self.optimizerP.zero_grad()
            loss_pi.backward()
            self.optimizerP.step()

            # train v_pi
            loss_v = F.smooth_l1_loss(self.V(s), r_cum)
            self.optimizerV.zero_grad()
            loss_v.backward()
            self.optimizerV.step()

            loss_pi_list.append(loss_pi.item())
            loss_v_list.append(loss_v.item())
            entropy_list.append(entropy.mean().item())

        return {'loss_discriminator': loss_d.item(),
                'loss_actor': np.mean(loss_pi_list),
                'loss_critic': np.mean(loss_v_list),
                'd_expert': d_expert.mean().item(),
                'd_agent': d_agent.mean().item(),
                'entropy': np.mean(entropy_list)}


