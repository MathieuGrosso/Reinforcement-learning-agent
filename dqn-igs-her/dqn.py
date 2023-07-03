import torch
import argparse
import sys

import gym
import gridworld
import datetime
from time import sleep

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import *



import numpy as np
from random import randint, random
from torch import nn

from collections import namedtuple

Transition = namedtuple('Transition',
                        ['state', 'action', 'goal',
                         'next_state', 'reward', 'done'])

class Memory:
    def __init__(self, buffer_size):
        self.buffer = np.zeros(buffer_size, dtype=Transition)
        self.buffer_size = buffer_size
        self.counter = 0
        self.is_full = False
        self.train_iteration = 0
        
    def store(self, transition):
        if self.is_full:
            idx = randint(0, self.buffer_size - 1)
            self.buffer[idx] = transition
        else:
            self.buffer[self.counter] = transition
            self.counter += 1
            if self.counter == self.buffer_size:
                self.is_full = True
        
    def sample(self, n_samples):
        if self.counter == 0:
            return None
        
        return np.random.choice(self.buffer[0:self.counter], n_samples, replace=True)
        
    
class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        
        self.nn = nn.Sequential(
                nn.Linear(input_size, 200),
                nn.Tanh(),
                nn.Linear(200, 200),
                nn.Tanh(),
                nn.Linear(200, output_size),
            )

    def forward(self, x):
        return self.nn(x)


def hard_update(target, source):
      target.load_state_dict(source.state_dict())


class Qnn:
    def __init__(self, input_size, output_size,
                 buffer_size, batch_size, copy_freq, lr):
        self.qnn = NN(input_size, output_size)
        self.qnn_target = NN(input_size, output_size)
        
        # Make sure target is with the same weight
        hard_update(self.qnn_target, self.qnn)
        
        self.memory = Memory(buffer_size)
        self.batch_size = batch_size
        self.copy_freq = copy_freq

        self.optim = torch.optim.Adam(self.qnn.parameters(), lr=lr)
        self.optim.zero_grad()

        self.criterion = nn.SmoothL1Loss()
        # self.criterion = nn.MSELoss()
        self.train_iteration = 0

    def push(self, l):
        self.memory.store(l)
    
    def train(self, gamma, t, logger):
        # Sample batch
        samples = self.memory.sample(self.batch_size)
        
        if samples is None:
            return
        
        states = torch.tensor([s.state for s in samples],
                              dtype=torch.float32).squeeze(1)
        actions = torch.tensor([s.action for s in samples],
                               dtype=torch.int64)
        goal = torch.tensor([s.goal for s in samples],
                            dtype=torch.float32).squeeze(1)
        next_states = torch.tensor([s.next_state for s in samples],
                                   dtype=torch.float32).squeeze(1)
        rewards = torch.tensor([s.reward for s in samples],
                               dtype=torch.float32)
        done = torch.tensor([s.done for s in samples],
                            dtype=torch.int64)
        
        # Compute loss and update
        Q = self.qnn.forward(
                torch.cat([states, goal], dim=1)
            ).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            Q_hat = self.qnn_target.forward(
                    torch.cat([next_states, goal], dim=1)
                ).max(dim=1).values.view(-1, 1)
            Q_hat = (Q_hat * (1 - done).view(-1, 1))

        self.optim.zero_grad()
        loss = self.criterion(Q.flatten(), (gamma * Q_hat.detach() + rewards.view(-1, 1)).flatten())
        
        loss.backward()
        self.optim.step()

        self.train_iteration += 1

        logger.add_scalar("loss",
                          loss.item(),
                          self.train_iteration)
        logger.add_scalar("distance_models",
                          self.distance_models(),
                          self.train_iteration)

        # Update target network if needed
        if t % self.copy_freq == 0:
            hard_update(self.qnn_target, self.qnn)

    def predict(self, couple):
        with torch.no_grad():
            return self.qnn.forward(couple)

    def distance_models(self):
         return sum(
             (x - y).abs().sum()
             for x, y in zip(self.qnn.state_dict().values(),
                             self.qnn_target.state_dict().values())
             )


def greedy(opt, action_space, **kwargs):
    """
    Sélectionne la meilleure action dans un dictionnaire
    `opt` du type `{action: estimated_rewards}`
    """
    if len(opt) == 0:
        return action_space.sample()
    
    return opt.argmax().item()

def egreedy(opt, action_space, eps):
    if random() > eps:
        return greedy(opt, action_space)
    else:
        return action_space.sample()

class DQN:
    def __init__(self, state_dim, action_space,
                 writer, featureExtractor,
                 gamma=0.99, batch_size=1000,
                 buffer_size=100000, copy_freq=1000,
                 train_freq=10, epsilon=0.2, lr=0.001):
        
        state_dim = state_dim
        action_dim = action_space.n
        
        self.writer = writer
        self.action_space = action_space
        self.featureExtractor = featureExtractor

        self.qnn = Qnn(2 * state_dim, action_dim, buffer_size, batch_size, copy_freq, lr=lr)
        self.train_freq = train_freq
        self.copy_freq = copy_freq

        self.gamma = gamma
        self.epsilon = epsilon

        self.s_t = None 
        self.a_t = None 
        
        self.t = 0
    
    def chose(self, s, goal):
        couple = torch.tensor([s, goal], dtype=torch.float).flatten()
        A = self.qnn.predict(couple)
        return egreedy(A, self.action_space, self.epsilon)

    def act(self, ob, goal, reward, done):
        ob = self.featureExtractor.getFeatures(ob)
        
        action = self.chose(ob, goal)
        
        if not self.a_t is None:
            transition = Transition(self.s_t, self.a_t, goal,
                                    ob, reward, done)
            self.qnn.push(transition)
        
        self.s_t = ob
        self.a_t = action
    
        if self.t % self.train_freq == 0:
            self.qnn.train(self.gamma, self.t, self.writer)
        self.t += 1

        return action

    def reset(self, s0):
        self.a_t = None
        self.s_t = self.featureExtractor.getFeatures(s0)




def main():
    config_name = "random_gridworld_Easy"
    config = load_yaml(f'./configs/config_{config_name}.yaml')

    
    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])
        
    max_episodes = config["max_episodes"]
    max_timesteps = config["max_timesteps"]
    
    gamma = config["gamma"]
    lr = config["learning_rate"]
    batch_size = config["batch_size"]
    buffer_size = config["buffer_size"]
    copy_freq = config["copy_freq"]
    train_freq = config["train_freq"]
    epsilon = config["epsilon"]
    
    s0 = env.reset()
    
    featureExtractor = config.featExtractor(env)
    s0 = featureExtractor.getFeatures(s0)
    state_dim = s0.shape[1]
    

    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"runs/MAP2/{config_name}/{dt}")

    agent = DQN(state_dim,
                env.action_space,
                writer,
                featureExtractor,
                gamma=gamma,
                batch_size=batch_size,
                buffer_size=buffer_size,
                copy_freq=copy_freq,
                train_freq=train_freq,
                epsilon=epsilon,
                lr=lr)
    
    for i_episode in range(0, max_episodes+1):
        done = False
        ob = env.reset()
        agent.reset(ob)
        
        goal, _ = env.sampleGoal()
        goal = featureExtractor.getFeatures(goal)
        cum_reward = 0
        reward = -0.1
        if i_episode % 1000 == 0:
            cum_reward_curve = 0

            
        for t in range(max_timesteps):
            action = agent.act(ob, goal, reward, done)
            ob, _, _, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(ob)
            done = (new_ob == goal).all()
            reward = 1.0 if done else -0.1

            cum_reward += reward

            if done:
                action = agent.act(ob, goal, reward, done)
                break
                
        cum_reward_curve += cum_reward
        if i_episode+1 % 1000 == 0:
            writer.add_scalar("mean_cum_reward", cum_reward_curve/1000, (i_episode+1)//1000)

        writer.add_scalar("length", t, i_episode)
        writer.add_scalar("reward", cum_reward, i_episode)
        print(f'Cum Reward: {cum_reward}, episode n°{i_episode}\n')

    env.close()

if __name__ == '__main__':
    main()