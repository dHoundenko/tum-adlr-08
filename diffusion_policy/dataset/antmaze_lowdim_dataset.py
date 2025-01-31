from typing import Dict
import torch
import numpy as np
import copy
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.pytorch_util import dict_apply

class AntMazeLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            obs_key='keypoint',   # 假设你的观测存在某个key
            action_key='action',
            # 如果antmaze还有其他信息，比如episode区间，可以在这里加
            ):
        """
        仿照pushT写法：
        1. 用ReplayBuffer读取zarr
        2. 用get_val_mask来划分train/val
        3. 用SequenceSampler做时序采样
        """
        super().__init__()
        # 1. 从zarr创建 ReplayBuffer
        #   如果你的antmaze数据并没有多条episode的概念，需要你自行封装
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, action_key]  # 你还有其他key就加上
        )

        # 2. 根据val_ratio, seed来划分train/val
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask

        # 如果你想限制训练的episode数量
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        # 3. 用SequenceSampler
        #   horizon表示一次采样多少步
        #   pad_before/after决定是否在序列开头或结尾补零之类
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        # 保存字段
        self.obs_key = obs_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.zarr_path = zarr_path  # 以防后面需要

    def get_validation_dataset(self):
        """
        和 pushT 一样，复制当前对象，让 sampler 用 val_mask
        """
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """
        仿照 pushT，用 _sample_to_data 把整份 replay_buffer 的数据转成 (obs, action)，
        再做 LinearNormalizer.fit
        """
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """
        与 pushT 类似，返回所有动作
        """
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        """
        由 SequenceSampler 决定可以采多少序列
        """
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        仿照 pushT, 从 sampler 取出 idx 对应的一段序列
        """
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def _sample_to_data(self, sample):
        """
        pushT 是把 keypoint + agent_pos 拼到 obs
        antmaze 你可以按自己需要组装
        """
        obs = sample[self.obs_key]   # (T, obs_dim)
        act = sample[self.action_key]# (T, act_dim)
        # 如果还需要额外拼 goal、agent_pos 等，也可在这里做
        data = {
            'obs': obs, 
            'action': act,
        }
        return data