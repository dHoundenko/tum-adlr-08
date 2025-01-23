# diffusion_policy/dataset/antmaze_lowdim_dataset.py
import os
import zarr
import numpy as np
import torch
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

class AntMazeLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
                 zarr_path='antmaze_medium_play_replay.zarr',
                 split='train',
                 train_ratio=0.9,
                 **kwargs):
        super().__init__()
        store = zarr.open(zarr_path, 'r')
        self.observations = store['data']['state'][:]
        self.actions = store['data']['action'][:]

        N = len(self.observations)
        train_size = int(N * train_ratio)
        if split == 'train':
            self.indices = np.arange(0, train_size)
        else:
            self.indices = np.arange(train_size, N)

        # 计算 mean/std 做normalizer
        obs_mean = self.observations[self.indices].mean(axis=0)
        obs_std  = self.observations[self.indices].std(axis=0) + 1e-6
        act_mean = self.actions[self.indices].mean(axis=0)
        act_std  = self.actions[self.indices].std(axis=0) + 1e-6

        self._normalizer = {
            'obs': LinearNormalizer(obs_mean, obs_std),
            'action': LinearNormalizer(act_mean, act_std)
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        obs = self.observations[i]
        act = self.actions[i]

        # (T, obs_dim), (T, act_dim) - 简单设T=1
        obs = obs[None,:]
        act = act[None,:]

        return {
            'obs': torch.from_numpy(obs),
            'action': torch.from_numpy(act)
        }

    def get_normalizer(self):
        return self._normalizer

    def get_validation_dataset(self):
    # 复用 self.__init__ 里用过的 zarr_path
    return AntMazeLowdimDataset(
        zarr_path='antmaze_medium_play_replay.zarr',
        split='val',
    )