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
        # For discrete actions, store action index (1,) instead of one-hot (action_space,)
        self.actions = th.empty((self.capacity, self.action_space), dtype=th.float32, device=self.device)
        self.rewards = th.empty((self.capacity, 1), dtype=th.float32, device=self.device)
        self.dones = th.empty((self.capacity, 1), dtype=th.float32, device=self.device)
        self.is_first = th.empty((self.capacity, 1), dtype=th.float32, device=self.device)

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
        """Sample sequences from the replay buffer.
        
        Samples batch_size sequences of length batch_length.
        Start indices are sampled uniformly, ensuring sequences don't wrap around.
        """
        max_idx = self.capacity if self.full else self.ptr
        if max_idx < batch_length:
            print(f"Buffer too small: {max_idx} < {batch_length}")
            return None
        
        # Sample start indices for sequences
        # Ensure we don't sample indices that would cause wrap-around
        max_start_idx = max_idx - batch_length
        if max_start_idx <= 0:
            return None
            
        start_indices = th.randint(0, max_start_idx, (batch_size,), device=self.device)
        
        # Create sequence indices
        batch_indices = start_indices.unsqueeze(1) + th.arange(batch_length, device=self.device).unsqueeze(0)
        batch_indices = batch_indices.flatten()
        
        states = self.states[batch_indices].reshape(batch_length, batch_size, -1)
        actions = self.actions[batch_indices].reshape(batch_length, batch_size, -1)
        rewards = self.rewards[batch_indices].reshape(batch_length, batch_size, -1)
        dones = self.dones[batch_indices].reshape(batch_length, batch_size, -1)
        is_first = self.is_first[batch_indices].reshape(batch_length, batch_size, -1)

        return RolloutBufferSamples(
            states,
            actions,
            rewards,
            dones,
            is_first
        )