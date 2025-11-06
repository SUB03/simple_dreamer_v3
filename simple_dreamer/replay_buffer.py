from typing import NamedTuple

from collections import deque

import torch as th
import numpy as np


class RolloutBufferSamples(NamedTuple):
    states: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    dones: th.Tensor
    is_first: th.Tensor

class ReplayBuffer:
    def __init__(self, config, obs_shape, action_space, device):
        self.capacity = int(config.capacity)
        self.device = device
        self.obs_shape = obs_shape
        self.action_space = action_space

        self.states = th.empty((self.capacity, self.obs_shape), dtype=th.float32, device=self.device)
        self.actions = th.empty((self.capacity, self.action_space), dtype=th.float32, device=self.device)
        self.rewards = th.empty(self.capacity, dtype=th.float32, device=self.device)
        self.dones = th.empty(self.capacity, dtype=th.int8, device=self.device)
        self.is_first = th.empty(self.capacity, dtype=th.int8, device=self.device)

        self.online_queue = deque(maxlen=self.capacity) # stores indices of recent transitions
        self.ptr = 0
        self.full = False
        
    def add(self, obs, action, reward, done, is_first):
        self.states[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.is_first[self.ptr] = is_first

        self.online_queue.append(self.ptr)

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True
    
    def sample(self, batch_length, batch_size):
        """each minibatch is formed first from non-overlapping online trajectories and then filled
            up with uniformly sampled trajectories from the replay buffer. """
        
        max_idx = self.capacity if self.full else self.ptr
        if max_idx < batch_length * batch_size:
            print(max_idx)
            return
        
        online_indices = []
        num_online = min(batch_length * batch_size, len(self.online_queue))
        for _ in range(num_online):
            online_indices.append(self.online_queue.popleft())
        
        remaining_needed = batch_length * batch_size - len(online_indices)

        uniform_indices = []
        if remaining_needed > 0:
            if self.full:
                uniform_indices = th.randint(0, self.capacity, (remaining_needed, ), device=self.device)
            else:
                uniform_indices = th.randint(0, self.ptr, (remaining_needed, ), device=self.device)
            uniform_indices = uniform_indices.tolist()

        idx = online_indices + uniform_indices

        states = self.states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]
        is_first = self.is_first[idx]

        return RolloutBufferSamples(
            states.reshape(batch_length, batch_size, -1),
            actions.reshape(batch_length, batch_size, -1),
            rewards.reshape(batch_length, batch_size, -1),
            dones.reshape(batch_length, batch_size, -1),
            is_first.reshape(batch_length, batch_size, -1)
        )