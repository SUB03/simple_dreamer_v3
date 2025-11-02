import math
import random
from collections import deque, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import gymnasium as gym

H, W, C = 64, 64, 4            # observation size (resize your frames to this)
LATENT_DIM = 64                # stochastic latent dim
DETERMINISTIC_SIZE = 256      # deterministic RNN state size
RSSM_INPUTS = LATENT_DIM + 32  # example action embedding size included
IMAGINATION_HORIZON = 15
SEQ_LEN = 50                   # training sequence length
BATCH_SIZE = 16
REPLAY_CAPACITY = 200_000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LR = 3e-4
MODEL_UPDATES_PER_COLLECT = 200
ACTOR_CRITIC_UPDATES_PER_COLLECT = 800
REPLAY_SAMPLES_PER_UPDATE = 1

class Encoder(nn.Module):
    def __init__(self, out_dim=LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.SiLU(),
        )
        self.fc = nn.Linear(256 * (H // 16) * (W // 16), out_dim)

    def forward(self, x):
        x = x.float() / 255.0
        x = x.permute(0,3,1,2)  # NHWC -> NCHW
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

class Decoder(nn.Module):
    def __init__(self, in_dim=LATENT_DIM + DETERMINISTIC_SIZE):
        super().__init__()
        flat = 256 * (H // 16) * (W // 16)
        self.fc = nn.Linear(in_dim, flat)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.SiLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.SiLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.SiLU(),
            nn.ConvTranspose2d(32,C,4,2,1),
        )
    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, H // 16, W // 16)
        x = self.deconv(h)
        return x  # logits (not activated)

# -----------------------------
# RSSM: simple deterministic + stochastic block
# -----------------------------
class RSSM(nn.Module):
    def __init__(self, action_size, deter_size=DETERMINISTIC_SIZE, stoch_size=LATENT_DIM):
        super().__init__()
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.action_embed = nn.Linear(action_size, 32)
        self.obs_embed = nn.Linear(LATENT_DIM, 128)
        self.gru = nn.GRUCell(128 + 32, deter_size)
        self._prior = nn.Linear(deter_size, 2 * stoch_size)
        self._post = nn.Linear(deter_size + 128, 2 * stoch_size)

    def init_state(self, batch_size):
        return {
            'deter': torch.zeros(batch_size, self.deter_size, device=DEVICE),
            'stoch': torch.zeros(batch_size, self.stoch_size, device=DEVICE)
        }

    def obs_step(self, prev_state, action, embed):
        a = self.action_embed(action)
        x = torch.cat([embed, a], dim=-1)
        deter = self.gru(x, prev_state['deter'])
        stats = self._post(torch.cat([deter, embed], dim=-1))
        mean, logstd = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(logstd) + 1e-5
        stoch = mean + std * torch.randn_like(mean)
        return {'deter': deter, 'stoch': stoch, 'mean': mean, 'std': std}

    def img_step(self, prev_state, action):
        a = self.action_embed(action)
        x = torch.cat([prev_state['stoch'], a], dim=-1)
        deter = self.gru(x, prev_state['deter'])
        stats = self._prior(deter)
        mean, logstd = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(logstd) + 1e-5
        stoch = mean + std * torch.randn_like(mean)
        return {'deter': deter, 'stoch': stoch, 'mean': mean, 'std': std}

# -----------------------------
# Actor / Critic / Reward predictor
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, out_dim)
        )
    def forward(self,x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, in_dim, action_dim):
        super().__init__()
        self.net = MLP(in_dim, 2 * action_dim)
        self.action_dim = action_dim
    def forward(self, x):
        stats = self.net(x)
        mean, logstd = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(logstd) + 1e-5
        return mean, std

class Critic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = MLP(in_dim, 1)
    def forward(self, x):
        return self.net(x).squeeze(-1)

class RewardPredictor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = MLP(in_dim, 1)
    def forward(self,x):
        return self.net(x).squeeze(-1)

Transition = namedtuple('Transition', ['obs','action','reward','done'])
class SequenceReplay:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push_episode(self, episode):
        # episode: list of Transition
        self.buffer.append(episode)

    def sample_batch(self, batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
        batch = []
        for _ in range(batch_size):
            ep = random.choice(self.buffer)
            if len(ep) >= seq_len:
                start = random.randint(0, len(ep) - seq_len)
                seg = ep[start:start+seq_len]
            else:
                # pad by repeating last
                seg = ep + [ep[-1]] * (seq_len - len(ep))
            batch.append(seg)
        # convert to tensors
        obs = torch.stack([torch.from_numpy(np.stack([t.obs for t in s])).to(DEVICE) for s in batch])
        actions = torch.stack([torch.from_numpy(np.stack([t.action for t in s])).to(DEVICE) for s in batch])
        rewards = torch.stack([torch.tensor([t.reward for t in s], dtype=torch.float32, device=DEVICE) for s in batch])
        dones = torch.stack([torch.tensor([t.done for t in s], dtype=torch.float32, device=DEVICE) for s in batch])
        return obs, actions, rewards, dones

    def __len__(self):
        return len(self.buffer)

def gaussian_kl(mean1, std1, mean2, std2):
    var1 = std1.pow(2)
    var2 = std2.pow(2)
    return ((mean1 - mean2).pow(2) / (2*var2) + 0.5*(var1/var2 - 1) - torch.log(std1/std2)).sum(dim=-1)


def lambda_returns(rewards, values, bootstrap, discount=0.99, lam=0.95):
    # rewards, values are (T,)
    next_values = torch.cat([values[1:], bootstrap.unsqueeze(0)], dim=0)
    deltas = rewards + discount * next_values - values
    adv = torch.zeros_like(deltas)
    last = 0
    for t in reversed(range(len(deltas))):
        last = deltas[t] + discount * lam * last
        adv[t] = last
    returns = adv + values
    return returns, adv

class DreamerTrainer:
    def __init__(self, env, action_size):
        self.env = env
        self.action_size = action_size
        self.encoder = Encoder().to(DEVICE)
        self.decoder = Decoder().to(DEVICE)
        self.rssm = RSSM(action_size).to(DEVICE)
        self.reward_predictor = RewardPredictor(LATENT_DIM + DETERMINISTIC_SIZE).to(DEVICE)
        self.actor = Actor(LATENT_DIM + DETERMINISTIC_SIZE, action_size).to(DEVICE)
        self.critic = Critic(LATENT_DIM + DETERMINISTIC_SIZE).to(DEVICE)

        self.model_opt = optim.Adam(list(self.encoder.parameters()) + list(self.rssm.parameters()) + list(self.decoder.parameters()) + list(self.reward_predictor.parameters()), lr=LR)
        self.ac_opt = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=LR)

        self.replay = SequenceReplay()

    def collect_episode(self, policy, max_steps=500):
        obs, _ = self.env.reset()
        episode = []
        done = False
        steps = 0
        while not done and steps < max_steps:
            with torch.no_grad():
                ob_t = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)
                embed = self.encoder(ob_t)
                # use RSSM to get state (very simplified: use embed as stochastic)
                action = policy(ob_t)
            next_obs, reward, done, _ = self.env.step(action)
            episode.append(Transition(obs, action, reward, done))
            obs = next_obs
            steps += 1
        self.replay.push_episode(episode)
        return episode

    def train_world_model(self, epochs=1):
        if len(self.replay) < 1:
            return
        for _ in range(epochs):
            obs, actions, rewards, dones = self.replay.sample_batch()
            B, T, H0, W0, C0 = obs.shape
            obs_flat = obs.view(B*T, H0, W0, C0)
            embeds = self.encoder(obs_flat)
            embeds = embeds.view(B, T, -1)
            # simple teacher-forced RSSM encoding along time
            prev = self.rssm.init_state(B)
            prior_means, prior_stds, post_means, post_stds = [],[],[],[]
            deter_states = []
            for t in range(T):
                act = actions[:,t].float()
                post = self.rssm.obs_step(prev, act, embeds[:,t])
                prior = self.rssm.img_step(prev, act)
                prior_means.append(prior['mean']); prior_stds.append(prior['std'])
                post_means.append(post['mean']); post_stds.append(post['std'])
                deter_states.append(post['deter'])
                prev = post
            # stack
            prior_means = torch.stack(prior_means, dim=1)
            prior_stds = torch.stack(prior_stds, dim=1)
            post_means = torch.stack(post_means, dim=1)
            post_stds = torch.stack(post_stds, dim=1)
            deter = torch.stack(deter_states, dim=1)
            stoch = post_means  # sample was used earlier; use mean for loss
            flat_feat = torch.cat([stoch, deter[:, -1]], dim=-1)
            recon = self.decoder(flat_feat)
            # reconstruction loss on the last frame (toy)
            target = obs[:, -1].float() / 255.0
            recon_loss = F.mse_loss(torch.sigmoid(recon), target)
            # KL
            kl_loss = gaussian_kl(post_means, post_stds, prior_means, prior_stds).mean()
            # reward pred loss
            reward_pred = self.reward_predictor(flat_feat).squeeze(-1)
            reward_loss = F.mse_loss(reward_pred, rewards.mean(dim=1))
            loss = recon_loss + 1.0 * kl_loss + reward_loss
            self.model_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.rssm.parameters()), 100.0)
            self.model_opt.step()

    def imagine_and_update_actor_critic(self):
        # sample seeds from replay, get posterior states, then imagine forward using actor
        if len(self.replay) < 1:
            return
        obs, actions, rewards, dones = self.replay.sample_batch(batch_size= BATCH_SIZE, seq_len=SEQ_LEN)
        B, T, H0, W0, C0 = obs.shape
        obs_flat = obs.view(B*T, H0, W0, C0)
        embeds = self.encoder(obs_flat).view(B, T, -1)

        # get final posterior states as seeds
        prev = self.rssm.init_state(B)
        for t in range(T):
            act = actions[:,t].float()
            prev = self.rssm.obs_step(prev, act, embeds[:,t])
        seed_deter = prev['deter']; seed_stoch = prev['stoch']

        # imagine forward
        imagined_feats = []
        deter = seed_deter; stoch = seed_stoch
        for h in range(IMAGINATION_HORIZON):
            feat = torch.cat([stoch, deter], dim=-1)
            mean, std = self.actor(feat)
            action = mean  # deterministic for imagination; consider add noise
            next_state = self.rssm.img_step({'deter': deter, 'stoch': stoch}, action)
            deter, stoch = next_state['deter'], next_state['stoch']
            imagined_feats.append(torch.cat([stoch, deter], dim=-1))
        imagined_feats = torch.stack(imagined_feats, dim=1)  # B x H x D

        # predict rewards and values
        flat = imagined_feats.view(-1, imagined_feats.size(-1))
        rewards_pred = self.reward_predictor(flat).view(B, IMAGINATION_HORIZON)
        values = self.critic(flat).view(B, IMAGINATION_HORIZON)
        # bootstrap with last value
        bootstrap = values[:, -1]
        returns, adv = lambda_returns(rewards_pred.transpose(0,1), values.transpose(0,1), bootstrap)
        returns = returns.transpose(0,1).reshape(-1)
        adv = adv.transpose(0,1).reshape(-1)

        # actor loss: maximize value (or advantage-weighted)
        # simple policy-gradient style surrogate
        mean, std = self.actor(flat.detach())
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(mean).sum(-1)  # using deterministic action; placeholder
        actor_loss = -(values.detach().reshape(-1) * logp).mean()

        # critic loss
        value_pred = self.critic(flat)
        critic_loss = F.mse_loss(value_pred, returns.detach())

        self.ac_opt.zero_grad()
        (actor_loss + critic_loss).backward()
        self.ac_opt.step()

    def train(self, total_episodes=1000, steps_per_episode=2000):
        for ep in range(total_episodes):
            # collect 1 episode (single env)
            # use current actor policy (stochastic) for exploration
            self.collect_episode(lambda ob: np.zeros(self.action_size), max_steps=steps_per_episode)
            # train world model a lot
            self.train_world_model(epochs=MODEL_UPDATES_PER_COLLECT)
            # many actor/critic updates using imagination
            for _ in range(ACTOR_CRITIC_UPDATES_PER_COLLECT):
                self.imagine_and_update_actor_critic()
            if ep % 10 == 0:
                print(f"Ep {ep} - replay size: {len(self.replay)}")


if __name__ == '__main__':
    env = gym.make("LunarLander-v3")
    print(env.action_space)
    trainer = DreamerTrainer(env, env.action_space.n)
    trainer.train()
