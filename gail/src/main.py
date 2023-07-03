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
from memory import  Memory
from Agents import GAIL, ExpertAgent,BC


def test(agent, episodes: int, env: gym.Env):
    r = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        r_sum = 0
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs.copy()
            r_sum += reward
        r.append(r_sum)
    return r



def train_behavioral_cloning(conf, agent: BC, train_dataloader: DataLoader, env: gym.Env,
                             writer: SummaryWriter):
    global_step = 0
    for epoch in tqdm(range(conf.epochs)):
        for states, actions in train_dataloader:
            loss = agent.learn(states=states, actions=actions)
            global_step += 1

            if global_step % conf.log_every_n_steps == 0:
                writer.add_scalar('train_loss', loss, global_step)

            if global_step % conf.test_freq == 0:
                rewards = test(agent, conf.test_episodes, env)
                writer.add_scalar('reward', np.mean(rewards), epoch)


def train_gail(conf, agent: GAIL, env: gym.Env, train_dataloader: DataLoader, writer: SummaryWriter):
    global_step = 0
    for epoch in tqdm(range(conf.epochs)):
        writer.add_scalar('epoch', epoch, global_step)
        memory = Memory(conf.batch_size,conf.min_clip,conf.max_clip)
        # collect batch_size transitions from trajectories of the current policy
        while len(memory) < conf.train_every_n_events:
            obs = env.reset()
            done = False
            while not done:
                # act and observe the next step
                a = agent.act(obs)
                s_next, reward, done, info = env.step(a)
                memory.store(transition=(obs, a, reward, s_next, done, info))
                obs = s_next.copy()

        for i in range(conf.train_iter):
            memory.compute_cumulated_r(agent, agent.dim_a)
            loss_dict = agent.learn(batch_agent=memory.get_minibatch_proxy_reward(),
                                    batch_expert=next(iter(train_dataloader)))
            global_step += 1
            if global_step % conf.log_every_n_steps == 0:
                for key, value in loss_dict.items():
                    writer.add_scalar(key, value, global_step)

        if global_step % conf.test_freq == 0:
            rewards = test(agent, conf.test_episodes, env)
            writer.add_scalar('reward', np.mean(rewards), global_step)


def main():
    project_dir = Path(__file__).resolve().parents[0]
    conf = OmegaConf.load(project_dir.joinpath('config.yaml'))
    time_tag = datetime.datetime.now().strftime(f'%Y%m%d-%H%M%S')
    log_dir = f'runs/tp12-{conf.algo}-{time_tag}-seed{conf.seed}'
    writer = SummaryWriter(log_dir)
    save_src_and_config(log_dir, conf, writer)

    # seed the env
    seed_everything(conf.seed)
    env = gym.make('LunarLander-v2')
    env.seed(conf.seed)
    env.action_space.seed(conf.seed)
    print(f'{log_dir}')
    print(f'Training agent with conf :\n{OmegaConf.to_yaml(conf)}')

    # dimensions of the env
    dim_s = 8
    dim_a = 4

    expert = ExpertAgent(dim_a, dim_s, project_dir / 'expert.pkl')

    train_dataset = TensorDataset(expert.states, expert.actions)
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size)

    if conf.algo == 'bc':
        agent = BC(dim_s=dim_s, dim_a=dim_a, lr=conf.lr)
        train_behavioral_cloning(conf, agent, train_dataloader, env, writer)
    elif conf.algo == 'gail':
        agent = GAIL(dim_s=dim_s, dim_a=dim_a, lr=conf.lr, K=conf.K, clip_eps=conf.clip_eps,
                     entropy_weight=conf.entropy_weight)
        train_gail(conf, agent, env, train_dataloader, writer)


def save_src_and_config(log_dir: str, conf, writer):
    # save config and source files as text files
    with open(f'{log_dir}/conf.yaml', 'w') as f:
        OmegaConf.save(conf, f)
    for f in glob.iglob('*.py'):
        shutil.copy2(f, log_dir)

    conf_clean = {k: str(v) for (k, v) in conf.items()}
    writer.add_hparams(conf_clean, metric_dict={'score': 0.})



main()