import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import OneHotCategoricalStraightThrough
from torch.distributions.utils import probs_to_logits

from simple_dreamer import utils

class Critic(nn.Module):
    def __init__(self, config, latent_state_size):
        self.config = config
        self.latent_state_size = latent_state_size
        super(Critic, self).__init__()
        self.critic = utils.build_network(
            num_layers=3,
            input_dim=latent_state_size,
            hidden_dim=256,
            output_dim=config.bins,
            layer_norm=nn.LayerNorm,
            bias=False
        )

        self.optimizer = optim.Adam(self.parameters(), float(config.lr), betas=(config.b1, config.b2))
    
    def forward(self, x):
        return self.critic(x)


class SlowModel(nn.Module):
    def __init__(self, source, model, rate = 1.0, every = 1):
        super(SlowModel, self).__init__()
        self.source = source
        self.model = model
        self.rate = rate
        self.every = every
        self.register_buffer('count', th.tensor(0, dtype=th.int32))

        assert self.rate == 1 or self.rate < 0.5, self.rate
    
    def forward(self, x):
        return self.model(x)

    def update(self):
        self._init_once()

        if self.count % self.every == 0:
            mix = self.rate
        else:
            mix = 0.0

        source_state = self.source.state_dict()
        model_state = self.model.state_dict()

        # Update parameters with exponential moving average
        with th.no_grad():
            for key in model_state:
                if key in source_state:
                    source_param = source_state[key]
                    model_param = model_state[key]
                    
                    updated_param = mix * source_param + (1 - mix) * model_param
                    model_state[key].copy_(updated_param)
        
        self.count += 1

    def _init_once(self):
        if not any(self.model.parameters()):
            source_state_dict = self.source.state_dict()
            self.model.load_state_dict(source_state_dict)
            
        model_keys = set(self.model.state_dict().keys())
        source_keys = set(self.source.state_dict().keys())
        assert model_keys == source_keys, f"Model parameter mismatch: {model_keys} vs {source_keys}"