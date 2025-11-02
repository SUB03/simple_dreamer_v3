import torch as th
import numpy as np
from typing import NamedTuple

class TrajectoryBuffer:
    def __init__(self, config, obs_shape, action_space, device):
        self.capacity = int(config.capacity)
        self.device = device
        self.obs_shape = obs_shape
        self.action_space = action_space

        self.states = th.zeros((self.capacity, self.obs_shape), dtype=th.float32, device=self.device)
        self.actions = th.zeros((self.capacity, self.action_space), dtype=th.float32, device=self.device)
        self.rewards = th.zeros(self.capacity, dtype=th.float32, device=self.device)
        self.dones = th.zeros(self.capacity, dtype=th.int8, device=self.device)

        self.ptr = 0
        self.full = False
        
    def add(self, obs, action, reward, done):
        self.states[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size, batch_length):
        max_idx = self.capacity if self.full else self.ptr
        assert self.full or (max_idx - batch_length + 1 > batch_size),\
            "not enough samples"

        window = batch_length - 1
        dones_arr = self.dones[:max_idx]

        if self.full:
            ext = th.cat((dones_arr, dones_arr[:window]))
            prefix_sum = th.cat((th.zeros(1, dtype=dones_arr.dtype, device=dones_arr.device), ext.cumsum(0)))
            sums = prefix_sum[window:window+max_idx] - prefix_sum[:max_idx]
            valid_mask = (sums == 0)
            starts = th.nonzero(valid_mask).flatten().to(th.int32)

        else:
            prefix_sum = th.cat((th.zeros(1, dtype=dones_arr.dtype, device=dones_arr.device), dones_arr.cumsum(0)))
            sums = prefix_sum[window:max_idx] - prefix_sum[:max_idx-window]
            valid_mask = (sums == 0)
            starts = th.nonzero(valid_mask).flatten()

        if starts.numel() == 0:
            raise RuntimeError("No valid start indices available.")

        rand_idx = th.randperm(starts.size(0), device=self.device)
        starts = starts[rand_idx]

        return self._batch(starts, batch_size, batch_length, max_idx, self.states,\
            self.actions, self.rewards, self.dones)
    
    def _batch(self, starts: th.Tensor, batch_size, batch_length, max_idx, *data_tensors):
        total = starts.size(0)
        offsets = th.arange(batch_length, device=self.device)
        for i in range(0, total, batch_size):
            chunk = starts[i:i+batch_size]
            B = chunk.numel()
            idx = (chunk.unsqueeze(1) + offsets.unsqueeze(0)) % max_idx
            batch_tensors = []
            for t in data_tensors:
                flat = idx.reshape(-1)
                gathered = t.index_select(0, flat)
                shape = (batch_length, B) + t.shape[1:]
                batch_tensors.append(gathered.view(shape))
            yield RolloutBufferSamples(*batch_tensors)
    
    def sample_batch(self, batch_size, batch_length):
        max_idx = self.ptr - batch_length + 1
        assert self.full or (max_idx > batch_size), "not enough data in the buffer to sample"

        sample_idx = th.randint(0, self.capacity if self.full else max_idx, (batch_size, ), device=self.device).reshape(-1, 1)
        batch_length = th.arange(batch_length, device=self.device).reshape(1, -1)

        sample_idx = (sample_idx + batch_length) % self.capacity

        states = self.states[sample_idx]
        actions = self.actions[sample_idx]
        rewards = self.rewards[sample_idx]
        dones = self.dones[sample_idx]

        return RolloutBufferSamples(
            states.transpose(1, 0),
            actions.transpose(1, 0),
            rewards.transpose(1, 0),
            dones.transpose(1, 0)
        )
    
    # samples based on start indicies and window
    def smart_batch(self, batch_size, batch_length, batch_first=False):
        max_idx = self.ptr - batch_length + 1
        assert self.full or (max_idx > batch_size), "not enough data in the buffer to sample"
        swap = lambda x: x.transpose(1, 0) if batch_first != True else x

        if self.full:
            view_dones = th.cat([self.dones[self.ptr:], self.dones[:self.ptr]]).to(self.device)
            view_len = self.capacity
            base_idx = th.arange(self.ptr, self.ptr + view_len, device=self.device) % self.capacity
        else:
            view_dones = self.dones[:self.ptr].to(self.device)
            view_len = self.ptr
            base_idx = th.arange(0, view_len, device=self.device)

        # Not enough contiguous timesteps
        if view_len < batch_length:
            raise AssertionError("Not enough contiguous timesteps to form one batch_length sequence.")

        windows = view_dones.unfold(0, batch_length, 1)  # shape (num_windows, batch_length)
        num_windows = windows.shape[0]

        if batch_length > 1:
            invalid = windows[:, :-1].any(dim=1)
        else:
            invalid = th.zeros(num_windows, dtype=th.bool, device=self.device)

        valid_mask = ~invalid
        valid_starts = th.nonzero(valid_mask).flatten()  # positions in view (0..num_windows-1)

        if valid_starts.numel() == 0:
            raise AssertionError("No valid sequence start positions (all sequences would cross episode boundaries).")

        # Sample start positions
        replace = (valid_starts.numel() < batch_size)
        if replace:
            # sample with replacement
            idx_choice = th.randint(0, valid_starts.numel(), (batch_size,), device=self.device)
            chosen_view_starts = valid_starts[idx_choice]
        else:
            # sample without replacement
            perm = th.randperm(valid_starts.numel(), device=self.device)[:batch_size]
            chosen_view_starts = valid_starts[perm]

        # Build sample_idx shape (batch_size, batch_length)
        offsets = th.arange(batch_length, device=self.device).unsqueeze(0)  # (1, L)
        sample_view_idx = chosen_view_starts.unsqueeze(1) + offsets  # (batch_size, L)

        # Map view indices back to original buffer indices
        # base_idx maps view index -> original index, so:
        sample_idx = base_idx[sample_view_idx]  # (batch_size, L)

        # Index into buffers
        states = self.states[sample_idx]    # (batch, seq, obs)
        actions = self.actions[sample_idx]  # (batch, seq, act)
        rewards = self.rewards[sample_idx]  # (batch, seq)
        dones = self.dones[sample_idx]      # (batch, seq)

        return RolloutBufferSamples(
            swap(states),
            swap(actions),
            swap(rewards),
            swap(dones)
        )

    

class RolloutBufferSamples(NamedTuple):
    states: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    dones: th.Tensor


if __name__ == "__main__":
    pass