import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategoricalStraightThrough, Bernoulli, OneHotCategorical, kl_divergence

from simple_dreamer.utils import symlog, symexp

class Output:
    def __repr__(self):
        name = type(self).__name__
        pred = self.pred()
        return f'{name}({pred.dtype}, shape={pred.shape})'

    def pred(self):
        raise NotImplementedError

    def loss(self, target):
        return -self.log_prob(target.detach())

    def sample(self, seed, shape=()):
        raise NotImplementedError

    def log_prob(self, event):
        raise NotImplementedError

    def prob(self, event):
        return th.exp(self.logp(event))

    def entropy(self):
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError
    
# class Binary(Output):

#   def __init__(self, logit):
#     self.logit = logit.to(th.float32)

#   def pred(self) -> th.Tensor:
#     return (self.logit > 0)

#   def log_prob(self, event):
#     event = event.to(th.float32)
#     logp = th.nn.functional.logsigmoid(self.logit).squeeze(-1)
#     lognotp = th.nn.functional.logsigmoid(-self.logit).squeeze(-1)
#     return event * logp + (1 - event) * lognotp

#   def sample(self):
#     prob = th.nn.functional.logsigmoid(self.logit)
#     return Bernoulli(prob=prob).sample()

class BernoulliSafeMode(Bernoulli):
    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs, logits, validate_args)

    def mode(self):
        mode = (self.probs > 0.5).to(self.probs)
        return mode   
        
class TwoHot(Output):
    def __init__(self, logits, squash=None, unsquash=None):
        #logits = f32(logits)
        #assert logits.shape[-1] == len(bins), (logits.shape, len(bins))
        #assert bins.dtype == th.float32, bins.dtype
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)
        self.bins = th.linspace(-20, 20, steps=255, device=logits.device)
        self.squash = squash or (lambda x: x)
        self.unsquash = unsquash or (lambda x: x)
    
    @property
    def mode(self):
        return self.unsquash((self.probs * self.bins).sum(-1, keepdim=True))
    
    def pred(self) -> th.Tensor:
        # The naive implementation results in a non-zero result even if the bins
        # are symmetric and the probabilities uniform, because the sum operation
        # goes left to right, accumulating numerical errors. Instead, we use a
        # symmetric sum to ensure that the predicted rewards and values are
        # actually zero at initialization.
        # return self.unsquash((self.probs * self.bins).sum(-1))
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = self.probs[..., :m]
            p2 = self.probs[..., m: m + 1]
            p3 = self.probs[..., m + 1:]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m: m + 1]
            b3 = self.bins[..., m + 1:]
            wavg = (p2 * b2).sum(-1) + (th.flip((p1 * b1), [-1]) + (p3 * b3)).sum(-1)
            return self.unsquash(wavg)
        else:
            p1 = self.probs[..., :n // 2]
            p2 = self.probs[..., n // 2:]
            b1 = self.bins[..., :n // 2]
            b2 = self.bins[..., n // 2:]
            wavg = ((p1 * b1)[..., ::-1] + (p2 * b2)).sum(-1)
            return self.unsquash(wavg)

    def log_prob(self, target):
        assert target.dtype == th.float32, target.dtype
        target = self.squash(target)

        below = (self.bins <= target[..., None]).to(th.int32).sum(-1) - 1
        above = len(self.bins) - (self.bins > target[..., None]).to(th.int32).sum(-1)
        below = th.clip(below, 0, len(self.bins) - 1)
        above = th.clip(above, 0, len(self.bins) - 1)
        equal = (below == above)

        dist_to_below = th.where(equal, 1, th.abs(self.bins[below] - target))
        dist_to_above = th.where(equal, 1, th.abs(self.bins[above] - target))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, len(self.bins)) * weight_below[..., None] +
            F.one_hot(above, len(self.bins)) * weight_above[..., None])
        log_pred = F.log_softmax(self.logits, dim=-1)

        return (target * log_pred).sum(-1)
    
class SymlogDistribution:
    def __init__(
        self,
        mode: th.Tensor,
        dims: int,
        dist: str = "mse",
        agg: str = "sum",
        tol: float = 1e-8,
    ):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._dist = dist
        self._agg = agg
        self._tol = tol
        self._batch_shape = mode.shape[: len(mode.shape) - dims]
        self._event_shape = mode.shape[len(mode.shape) - dims :]

    @property
    def mode(self) -> th.Tensor:
        return symexp(self._mode)

    @property
    def mean(self) -> th.Tensor:
        return symexp(self._mode)

    def log_prob(self, value: th.Tensor) -> th.Tensor:
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2
            distance = th.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = th.abs(self._mode - symlog(value))
            distance = th.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss