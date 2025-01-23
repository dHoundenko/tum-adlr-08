import os
import numpy as np
import zarr
import gym
import d4rl

env = gym.make('antmaze-medium-play-v0')
data = d4rl.qlearning_dataset(env)

# data 包含:
# data['observations']: (N, state_dim)
# data['actions']: (N, action_dim)
# data['rewards']: (N, )
# data['terminals']: (N, )
# data['next_observations']: (N, state_dim)

observations = data['observations']  # (N, 27)
actions = data['actions']  # (N, 8)
terminals = data['terminals']

# 根据 terminals 来确定 episode_ends
episode_ends = np.where(terminals == 1)[0] + 1

if not os.path.exists("data"):
    os.makedirs("data")

store_path = "data/antmaze_medium_play_replay.zarr"
store = zarr.open(store_path, mode='w')
store.create_dataset('data/state', data=observations.astype(np.float32))
store.create_dataset('data/action', data=actions.astype(np.float32))

meta_group = store.create_group('meta')
meta_group.create_dataset('episode_ends', data=episode_ends.astype(np.int64))

print(f"Converted dataset saved to {store_path}")