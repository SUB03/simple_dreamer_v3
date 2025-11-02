import yaml, os
from attridict import AttriDict
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn

def findFile(filename):
    currentDir = os.getcwd()
    for root, dirs, files in os.walk(currentDir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"File '{filename}' not found in subdirectories of {currentDir}")

def loadConfig(config_path):
    config_path = findFile(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return AttriDict(config)

def log_losses(writer, loss_dict, step):
    for key, value in loss_dict.items():
        writer.add_scalar(key, value, step)

def symlog(x):
    return th.sign(x) * th.log1p(th.abs(x))

def symexp(x):
    return th.sign(x) * th.expm1(th.abs(x))

def probs_to_logits(probs):
    eps = th.finfo(probs.dtype).eps
    probs = th.clamp(probs, eps, 1-eps)
    return th.log(probs)

def init_xavier_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)

def init_xavier_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

def init_zero(m):
    if isinstance(m, nn.Linear):
        m.weight.data.zero_()
        m.bias.data.zero_()

def build_network(
    num_layers = 1,
    input_dim = None,
    hidden_dim = None,
    output_dim = 1,
    layer_norm = None,
    act = nn.SiLU,
    bias = True
) -> nn.Sequential:
    assert input_dim is not None, "Input dim cannot be None"
    assert num_layers > 1, num_layers
    assert hidden_dim is not None, "hidden dim should be not None for num_layers > 1"

    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
    if layer_norm:
            if layer_norm == nn.LayerNorm:
                layers.append(nn.LayerNorm(hidden_dim, 1e-3))
            else:
                raise NotImplementedError(layer_norm)
    layers.append(act())

    for _ in range(num_layers-2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        if layer_norm:
            if layer_norm == nn.LayerNorm:
                layers.append(nn.LayerNorm(hidden_dim, 1e-3))
            else:
                raise NotImplementedError(layer_norm)
        layers.append(act())
    
    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)

class Normalize(nn.Module):
    def __init__(self, smoothing = 0.99):
        super(Normalize, self).__init__()
        self.smoothing = smoothing
        self.perc_low = 0.05
        self.perc_high = 0.95
        self.register_buffer("low", th.zeros((), dtype=th.float32))
        self.register_buffer("high", th.zeros((), dtype=th.float32))

    def forward(self, x: th.Tensor):
        x = (x.to(th.float32)).detach()
        low = th.quantile(x, self.perc_low)
        high = th.quantile(x, self.perc_high)
        self.low = self.smoothing * self.low + (1 - self.smoothing) * low
        self.high = self.smoothing * self.high + (1 -self.smoothing) * high
        invscale = th.max(th.tensor(1e-8), self.high - self.low)
        return self.low.detach(), invscale.detach()
