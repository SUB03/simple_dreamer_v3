import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import kl_divergence, OneHotCategoricalStraightThrough, Independent
from torch.distributions.utils import probs_to_logits

from simple_dreamer import utils
import simple_dreamer.networks.outputs as outs

class WorldModel(nn.Module):
    def __init__(self, config, encoder_type, obs_shape, n_actions, device):
        super(WorldModel, self).__init__()
        self.config = config
        self.encoder_type = encoder_type
        self.unimix = config.unimix
        self.pred_weight = 1
        self.dynamic_weight = 0.5
        self.repr_weight = 0.1
        self.device = device
        self.C, self.D = config.latent_a, config.latent_b
        self.latent_shape = self.C*self.D # 32x32
        self.deter_size = config.recurrent_model.deter_size
        self.enc_size = 256
        self.bins = config.bins

        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 256, bias=False),
            nn.LayerNorm(256, eps=1e-3),
            nn.SiLU()
        )
        for m in self.encoder:
            utils.init_xavier_normal(m)
        utils.init_xavier_uniform(self.encoder[-1])

        self.decoder = utils.build_network(
            num_layers=2,
            input_dim=self.latent_shape + self.deter_size,
            hidden_dim=256,
            output_dim=obs_shape,
            layer_norm=nn.LayerNorm,
            act=nn.SiLU,
            bias=False
        )
        for m in self.decoder:
            utils.init_xavier_normal(m)

        self.post_net = utils.build_network(
            num_layers=3,
            input_dim=self.deter_size + self.enc_size,
            hidden_dim=256,
            output_dim=self.latent_shape,
            layer_norm=nn.LayerNorm,
            act=nn.SiLU,
            bias=False
        )
        for m in self.post_net:
            utils.init_xavier_normal(m)
        utils.init_xavier_uniform(self.post_net[-1])

        self.prior_net = utils.build_network(
            num_layers=3,
            input_dim=self.deter_size,
            hidden_dim=256,
            output_dim=self.latent_shape,
            layer_norm=nn.LayerNorm,
            act=nn.SiLU,
            bias=False
        )
        for m in self.prior_net:
            utils.init_xavier_normal(m)
        utils.init_xavier_uniform(self.prior_net[-1])

        self.reward_predictor = utils.build_network(
            num_layers=2,
            input_dim=self.deter_size + self.latent_shape,
            hidden_dim=256,
            output_dim=255,
            layer_norm=nn.LayerNorm,
            act=nn.SiLU,
            bias=False
        )
        for m in self.reward_predictor:
            utils.init_xavier_normal(m)
        utils.init_zero(self.reward_predictor[-1])

        self.continue_predictor = utils.build_network(
            num_layers=2,
            input_dim=self.deter_size + self.latent_shape,
            hidden_dim=256,
            layer_norm=nn.LayerNorm,
            act=nn.SiLU,
            bias=False
        )
        for m in self.continue_predictor:
            utils.init_xavier_normal(m)
        utils.init_xavier_uniform(self.continue_predictor[-1])

        self.recurrent_mlp = nn.Sequential(
            nn.Linear(self.latent_shape+n_actions, 256, bias=False),
            nn.LayerNorm(256, eps=1e-3),
            nn.SiLU()
        )
        for m in self.recurrent_mlp:
            utils.init_xavier_normal(m)

        self.recurrent_model = GRUCell(256, self.deter_size)

        self.optimizer = optim.Adam(self.parameters(), lr=float(config.world_model.lr)) 

    def encode(self, x) -> th.Tensor:
        x = utils.symlog(x) # symlog only if using MLP, not if CNN

        logits = self.encoder(x)
        return logits
    
    def decode(self, x, unsquash=None):
        output = self.decoder(x)
        return output
    
    
    def _reward_pred(self, x):
        # if self.bins % 2 == 1:
        #     half = th.linspace(-20, 0, (self.bins - 1) // 2 + 1, dtype=th.float32, device=self.device)
        #     half = utils.symexp(half)
        #     bins = th.concatenate([half, (-half[:-1]).flip(0)], 0)
        # else:
        #     half = th.linspace(-20, 0, self.bins // 2, dtype=th.float32, device=self.device)
        #     half = utils.symexp(half)
        #     bins = th.concatenate([half, (-half).flip(0)], 0)

        x = x.reshape(*x.shape[:2], -1)
        logits = self.reward_predictor(x)

        return outs.TwoHot(logits=logits, squash=utils.symlog, unsquash=utils.symexp)
    
    def _unimix(self, logits: th.Tensor) -> th.Tensor:
        dim = logits.dim()
        if dim == 3:
            logits = logits.view(*logits.shape[:-1], self.C, self.D)
        elif dim != 4:
            raise RuntimeError(f"The logits expected shape is 3 or 4: received a {dim}D tensor")
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            assert probs.shape[-1] == self.D, probs.shape
            uniform = th.ones_like(probs) / probs.shape[-1]
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = probs_to_logits(probs)
        return logits
    
    def get_prior(self, x: th.Tensor):
        logits = self.prior_net(x)
        logits = logits.reshape(*logits.shape[:-1], self.C, self.D)
        logits = self._unimix(logits)
        stoch_sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        logits = logits.reshape(*logits.shape[:-2], -1)
        return logits, stoch_sample 
    
    def get_post(self, recurrent_state: th.Tensor, embeddings: th.Tensor):
        x = th.cat((recurrent_state, embeddings), dim=-1)
        logits = self.post_net(x)
        logits = logits.reshape(*logits.shape[:-1], self.C, self.D)
        logits = self._unimix(logits)
        stoch_sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        logits = logits.reshape(*logits.shape[:-2], -1)
        return logits, stoch_sample
    
    def observe(self, recurrent_state: th.Tensor, post: th.Tensor, embedded_obs: th.Tensor, action: th.Tensor)\
        -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        
        post = post.reshape(*post.shape[:-2], -1)
        feat = self.recurrent_mlp(th.cat((post, action), dim=-1))
        recurrent_state = self.recurrent_model(feat, recurrent_state)
        prior_logits, _ = self.get_prior(recurrent_state)
        post_logits, post = self.get_post(recurrent_state, embedded_obs)

        return recurrent_state, post, post_logits, prior_logits

    def imagine(self, imagined_prior: th.Tensor, recurrent_state: th.Tensor, action: th.Tensor)\
        -> tuple[th.Tensor, th.Tensor]: 
        
        feat = self.recurrent_mlp(th.cat((imagined_prior, action), dim=-1))
        recurrent_state = self.recurrent_model(feat, recurrent_state)
        _, imagined_prior = self.get_prior(recurrent_state)

        return imagined_prior, recurrent_state

    def learn(self, batch):
        S, B, _ = batch.states.shape
        assert S == self.config.batch_length
        assert B == self.config.batch_size

        embed = self.encode(batch.states)

        # initialize
        recurrent_state = th.zeros((1, B, self.deter_size), dtype=th.float32, device=self.device)
        post = th.zeros((1, B, self.C, self.D), dtype=th.float32, device=self.device)
        batch_actions = th.cat((th.zeros_like(batch.actions[:1]), batch.actions[:-1]), dim=0)

        # store
        posts = th.empty((S, B, self.C, self.D), dtype=th.float32, device=self.device)
        recurrent_states = th.empty((S, B, self.deter_size), dtype=th.float32, device=self.device)
        posts_logits = th.empty((S, B, self.latent_shape), dtype=th.float32, device=self.device)
        priors_logits = th.empty((S, B, self.latent_shape), dtype=th.float32, device=self.device)

        for i in range(S):
            recurrent_state, post, post_logits, prior_logits = self.observe(
                recurrent_state,
                post,
                embed[i:i+1],
                batch_actions[i:i+1]
            )

            posts[i] = post
            recurrent_states[i] = recurrent_state
            posts_logits[i] = post_logits
            priors_logits[i] = prior_logits

        full_state = th.cat((posts.view(*posts.shape[:-2], -1), recurrent_states), dim=-1)
        reward_dist = self._reward_pred(full_state)
        continue_dist = Independent(outs.BernoulliSafeMode(logits=self.continue_predictor(full_state)), 1)
        continue_targets = 1.0 - batch.dones
        recon_logits = self.decode(full_state)
        recon_dist = outs.SymlogDistribution(recon_logits, 1)

        self.optimizer.zero_grad(set_to_none=True)
        reward_loss = -reward_dist.log_prob(batch.rewards)

        continue_loss = -continue_dist.log_prob(continue_targets.unsqueeze(-1))
        recon_loss = -recon_dist.log_prob(batch.states)

        posts_logits = posts_logits.view(*posts_logits.shape[:-1], self.C, self.D)
        priors_logits = priors_logits.view(*priors_logits.shape[:-1], self.C, self.D)
        dyn_loss = kl_divergence(
            Independent(OneHotCategoricalStraightThrough(logits=posts_logits.detach()), 1),
            Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1)
        )
        repr_loss = kl_divergence(
            Independent(OneHotCategoricalStraightThrough(logits=posts_logits), 1),
            Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1)
        )
        free_nats = th.full_like(dyn_loss, 1.0)
        dyn_loss = self.dynamic_weight * th.maximum(dyn_loss, free_nats)
        repr_loss = self.repr_weight * th.maximum(repr_loss, free_nats)

        pred_loss = recon_loss + reward_loss + continue_loss
        total_loss = (self.pred_weight * pred_loss + dyn_loss + repr_loss).mean()
        # print(f"total loss {total_loss.item()}, rwd_loss: {reward_loss.mean()}, obs_loss: {recon_loss.mean()} " +\
        #     f"cont_loss: {continue_loss.mean()}, dyn_loss: {dyn_loss.detach().mean()}, repr_loss: {repr_loss.detach().mean()}")

        total_loss.backward()
        th.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.optimizer.step()

        return {"total_loss": total_loss.item(), "rwd_loss": reward_loss.mean(), "rec_loss": recon_loss.mean(),\
            "cont_loss": continue_loss.mean(), "dyn_loss": dyn_loss.mean(), "repr_loss": repr_loss.mean()},\
            posts, recurrent_states

class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=th.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        parts = self.layers(th.cat([inputs, state], -1))
        reset, cand, update = th.split(parts, [self._size] * 3, -1)
        reset = th.sigmoid(reset)
        cand = self._act(reset * cand)
        update = th.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output#, [output]