import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Distribution, OneHotCategoricalStraightThrough
from torch.distributions.utils import probs_to_logits

from simple_dreamer import utils

class Actor(nn.Module):
    def __init__(self, config, latent_state_size, action_space, n_actions):
        super(Actor, self).__init__()
        self.actions_space = action_space
        self.n_actions = n_actions
        self.latent_state_size = latent_state_size
        self.unimix = config.unimix

        self.actor = utils.build_network(
            num_layers=3,
            input_dim=self.latent_state_size,
            hidden_dim=256,
            bias=False,
            layer_norm=nn.LayerNorm,
            output_dim=n_actions
        )
        for m in self.actor:
            utils.init_xavier_normal(m)

        self.optimizer = optim.Adam(self.parameters(),\
            float(config.actor.lr), eps=float(config.actor.eps)) # betas=(config.actor.b1, config.actor.b2)
    
    def _unimix(self, logits):
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = th.ones_like(probs) / probs.shape[-1]
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = probs_to_logits(probs)
        return logits
    
    def forward(self, x, greedy=False) -> tuple[th.Tensor, Distribution]:
        logits = self.actor(x)
        if self.actions_space == "continuous":
            raise NotImplementedError(self.actions_space)
        
        elif self.actions_space == "discrete":
            action_dist = OneHotCategoricalStraightThrough(logits=self._unimix(logits))
            if greedy:
                action = action_dist.mode
            else:
                action = action_dist.rsample()
        else:
            raise RuntimeError(self.actions_space)
        
        return action, action_dist
