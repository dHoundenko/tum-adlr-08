import gym
import torch
import numpy as np
import pathlib
import tqdm
import math
import imageio
import wandb

from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv

# if len(nobs.shape) == 2:
#             nobs = nobs.unsqueeze(1)  # now shape becomes (B, 1, obs_dim)
#         B, T_obs, Do = nobs.shape

class AntMazeLowdimRunner(BaseLowdimRunner):
    """
    An environment runner for the low-dimensional AntMaze task.
    
    This runner creates multiple instances of a Gym environment (e.g.
    "antmaze-medium-play-v0") using an asynchronous vectorized wrapper.
    It runs rollouts with the provided lowdim policy and collects rewards and rendered frames,
    which are then saved as videos.
    """
    def __init__(self,
                 output_dir,
                 env_name,          # e.g. "antmaze-medium-play-v0"
                 episodes,          # total number of evaluation episodes
                 max_steps,         # maximum steps per episode
                 render_hw,         # desired rendering resolution, e.g. [240, 360]
                 fps,               # frames per second for video recording
                 n_obs_steps,       # observation horizon (e.g. 2)
                 n_action_steps,    # action horizon (e.g. 8)
                 seed=0,
                 n_envs=None,      # number of parallel environments (default: episodes)
                 **kwargs):
        super().__init__(output_dir)
        self.env_name = env_name
        self.episodes = episodes
        self.max_steps = max_steps
        self.render_hw = render_hw
        self.fps = fps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.seed = seed

        if n_envs is None:
            n_envs = episodes
        self.n_envs = n_envs

        def env_fn():
            env = gym.make(self.env_name)
            # If the environment supports setting render size, do it here.
            if hasattr(env, "set_render_size"):
                env.set_render_size(*self.render_hw)
            # Optionally, you can seed the environment here:
            env.seed(self.seed)
            return env

        # Create a list of environment factory functions.
        self.env_fns = [env_fn for _ in range(self.n_envs)]
        # Instantiate the asynchronous vectorized environment.
        self.env = AsyncVectorEnv(self.env_fns)

    def run(self, policy):
        """
        Run evaluation rollouts for the given lowdim policy.
        
        Args:
            policy: A low-dimensional policy that implements:
                - reset() to reset its internal state.
                - predict_action(obs_dict) to return a dict containing key 'action'
                  as a torch.Tensor.
                - device attribute indicating the torch device.
        Returns:
            A dictionary of logged metrics, including per-episode maximum rewards
            and (if available) videos (wrapped via wandb.Video).
        """
        device = policy.device
        env = self.env
        n_envs = self.n_envs
        n_inits = self.episodes
        n_chunks = math.ceil(n_inits / n_envs)

        all_rewards = [None] * n_inits
        all_video_paths = [None] * n_inits  # to store video file paths

        # For evaluation, we simply run a rollout for each episode.
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            # Reset environments for this chunk.
            obs = env.reset()
            policy.reset()
            done = np.array([False] * n_envs)
            step = 0

            rewards_chunk = [[] for _ in range(n_envs)]
            frames_chunk = [[] for _ in range(n_envs)]

            with tqdm.tqdm(total=self.max_steps, desc=f"Rollout chunk {chunk_idx+1}/{n_chunks}") as pbar:
                while not np.all(done) and step < self.max_steps:
                    # Convert observations to torch tensor and send to the device.
                    obs_tensor = torch.from_numpy(obs).to(device)
                    with torch.no_grad():
                        action_tensor = policy.predict_action({'obs': obs_tensor})['action']
                    # Convert predicted actions to numpy.
                    action = action_tensor.cpu().numpy()
                    obs, reward, done, info = env.step(action)
                    step += 1
                    pbar.update(1)

                    for i in range(n_envs):
                        rewards_chunk[i].append(reward[i])

                    # Record frames if the environment supports rendering.
                    # We assume env.call('render') returns a list of RGB frames for each sub-environment.
                    rendered_frames = env.call('render')
                    for i in range(n_envs):
                        frames_chunk[i].append(rendered_frames[i])

            # Save rewards and videos for this chunk.
            for i in range(end - start):
                idx = start + i
                all_rewards[idx] = np.array(rewards_chunk[i])
                video_path = self._save_video(frames_chunk[i], episode=idx)
                all_video_paths[idx] = video_path

        # Reset the vectorized environment after rollouts.
        _ = env.reset()

        # Aggregate metrics.
        log_data = {}
        max_rewards = []
        for i in range(n_inits):
            max_r = np.max(all_rewards[i])
            max_rewards.append(max_r)
            log_data[f'episode_{i}_max_reward'] = max_r
            if all_video_paths[i] is not None:
                log_data[f'episode_{i}_video'] = wandb.Video(all_video_paths[i])
        log_data['mean_max_reward'] = np.mean(max_rewards)
        return log_data

    def _save_video(self, frames, episode):
        """
        Save a list of frames as an MP4 video using imageio.
        """
        video_dir = pathlib.Path(self.output_dir) / "media"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"antmaze_episode_{episode}.mp4"
        try:
            writer = imageio.get_writer(str(video_path), fps=self.fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            return str(video_path)
        except Exception as e:
            print(f"Error saving video for episode {episode}: {e}")
            return None
