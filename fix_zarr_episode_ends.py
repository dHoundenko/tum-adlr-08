import zarr
import sys

zarr_path = sys.argv[1] if len(sys.argv) > 1 else 'data/antmaze_medium_play_replay.zarr'
store = zarr.open(zarr_path, mode='a')  # 可写模式

last_episode_end = store['meta']['episode_ends'][-1]
n_samples = store['data']['state'].shape[0]
print(f"Before fix: episode_ends[-1]={last_episode_end}, data/state={n_samples}")

# 如果 episode_ends[-1] != n_samples，就修正:
if last_episode_end != n_samples:
    print(f"Fixing mismatch: setting episode_ends[-1] to {n_samples}")
    store['meta']['episode_ends'][-1] = n_samples

