import copy
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from simple_dreamer import utils
import simple_dreamer.networks.outputs as outs
from simple_dreamer.networks.actor import Actor
from simple_dreamer.networks.critic import Critic
from simple_dreamer.networks.world_model import WorldModel
from simple_dreamer.replay_buffer import ReplayBuffer

class DreamerV3:
    def __init__(self, config, encoder_type, obs_shape, n_actions, device):
        if type(obs_shape) == int:
            self.obs_shape = obs_shape
        elif type(obs_shape) == tuple:
            shape = 1
            for i in obs_shape:
                shape *= i
            self.obs_shape = shape
        else:
            raise NotImplementedError("Invalid type of shape")
        
        self.is_continuous = False
        self.recurrent_state_size = config.recurrent_model.deter_size
        self.latent_size = config.latent_a * config.latent_b
        self.full_state_size = self.latent_size + self.recurrent_state_size

        self.world_model = WorldModel(config, encoder_type, self.obs_shape, n_actions, device).to(device)
        self.actor = Actor(config, self.full_state_size, action_space="discrete", n_actions=n_actions).to(device)
        self.critic = Critic(config.critic, self.full_state_size).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.buffer = ReplayBuffer(config, self.obs_shape, n_actions, device)

        self.ema = utils.Normalize()

        print(self.world_model)
        print(self.actor)
        print(self.critic)

        self.device = device
        self.config = config
        self.n_actions = n_actions
        self.batch_size = config.batch_size
        self.batch_length = config.batch_length
    
    def add(self, obs, action, reward, done, is_first):
        self.buffer.add(obs, action, reward, done, is_first)

    def compute_lambda_values(self, rewards, values, continues, lmbda=0.95):
        vals = [values[-1:]]
        interm = rewards + continues * values * (1 - lmbda)
        for t in reversed(range(len(continues))):
            vals.append(interm[t] + continues[t] * lmbda * vals[-1])

        return th.cat(list(reversed(vals))[:-1])
    
    def learn(self, batch):
        # world model learning
        loss, posts, recurrent_states = self.world_model.learn(batch)

        # behavior training
        imagined_prior = posts.detach().reshape(1, -1, self.latent_size)
        recurrent_state = recurrent_states.detach().reshape(1, -1, self.recurrent_state_size)
        imagined_latent_state =  th.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories = th.empty(
            self.config.imagination_horizon + 1,
            self.batch_size * self.batch_length,
            self.full_state_size,
            device=self.device
        )
        imagined_trajectories[0] = imagined_latent_state
        imagined_actions = th.empty(
            self.config.imagination_horizon + 1,
            self.batch_size * self.batch_length,
            self.n_actions,
            device=self.device
        )

        actions = self.actor(imagined_latent_state.detach())[0]
        imagined_actions[0] = actions

        # imagine trajectories
        for i in range(1, self.config.imagination_horizon + 1):
            imagined_prior, recurrent_state = self.world_model.imagine(imagined_prior, recurrent_state, actions)
            imagined_prior = imagined_prior.reshape(1, -1, self.latent_size)
            imagined_latent_state = th.cat((imagined_prior, recurrent_state), -1)
            imagined_trajectories[i] = imagined_latent_state
            actions = self.actor(imagined_latent_state.detach())[0]
            imagined_actions[i] = actions

        predicted_values = outs.TwoHot(self.critic(imagined_trajectories)).mode
        predicted_rewards = outs.TwoHot(self.world_model.reward_predictor(imagined_trajectories)).mode
        continues = outs.BernoulliSafeMode(logits=self.world_model.continue_predictor(imagined_trajectories)).mode
        true_continues = (1 - batch.dones).flatten().reshape(1, -1, 1)
        continues = th.cat((true_continues, continues[1:]))

        lambda_values = self.compute_lambda_values(
            predicted_rewards[1:],
            predicted_values[1:],
            continues[1:] * self.config.gamma,
            lmbda=self.config.lmbda
        )
        with th.no_grad():
            discount = th.cumprod(continues * self.config.gamma, dim=0) / self.config.gamma
        
        self.actor.optimizer.zero_grad(set_to_none=True)
        _, policy_dist = self.actor.forward(imagined_trajectories.detach())

        baseline = predicted_values[:-1]
        offset, invscale = self.ema(lambda_values)
        normed_lambda_values = (lambda_values - offset) / invscale
        normed_baseline = (baseline - offset) / invscale
        advantage = normed_lambda_values - normed_baseline

        if self.is_continuous:
            objective = advantage
            raise NotImplementedError(self.is_continuous)
        else:
            # objective = th.stack(
            #     [
            #         policy.log_prob(imagined_action.detach()).unsqueeze(-1)[:-1]
            #         for policy, imagined_action in zip(policy_dist, th.split(imagined_actions, (self.n_actions, ), dim=-1))
            #     ],
            #     dim=-1
            # ).sum(-1) * advantage.detach()
        
            objective = policy_dist.log_prob(imagined_actions.detach()).unsqueeze(-1)[:-1]\
                * advantage.detach()
        
        entropy = float(self.config.actor.ent_coef) * policy_dist.entropy()
        actor_loss = -th.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(-1)[:-1]))
        actor_loss.backward()
        self.actor.optimizer.step()

        qv_dist = outs.TwoHot(self.critic(imagined_trajectories.detach()[:-1]),\
            squash=utils.symlog, unsquash=utils.symexp)
        tv = outs.TwoHot(self.target_critic(imagined_trajectories.detach()[:-1]),\
            squash=utils.symlog, unsquash=utils.symexp).mode

        self.critic.optimizer.zero_grad(set_to_none=True)
        value_loss = -qv_dist.log_prob(lambda_values.detach().squeeze(-1))
        value_loss = value_loss - qv_dist.log_prob(tv.detach().squeeze(-1))
        value_loss = th.mean(value_loss * discount[:-1].squeeze(-1))
        value_loss.backward()
        self.critic.optimizer.step()
        # print(f"policy_loss, {actor_loss.item()}, value_loss: {value_loss.item()}, entropy: {entropy.mean().item()}")
        loss.update({
            "policy_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().detach()
        })

        return loss
    
    @th.no_grad()
    def init_states(self, num_envs):
        action = th.zeros((1, 1, self.n_actions), dtype=th.float32, device=self.device)
        recurrent_state, latent_state = self.world_model.get_initial_states((1, num_envs))
        latent_state = latent_state.reshape(*latent_state.shape[:-2], -1)
        return recurrent_state, latent_state, action
    
    def sample_action(self, obs, recurrent_state, latent_state, action)\
            -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        
        embed = self.world_model.encode(obs).unsqueeze(0).unsqueeze(0)
        feat = self.world_model.recurrent_mlp(th.cat((latent_state, action), dim=-1))
        recurrent_state = self.world_model.recurrent_model(feat, recurrent_state)
        _, latent_state = self.world_model.get_post(recurrent_state, embed)
        latent_state = latent_state.reshape(*latent_state.shape[:-2], -1)
        action, _ = self.actor(th.cat((latent_state, recurrent_state), dim=-1))
        
        return recurrent_state, latent_state, action