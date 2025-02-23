# File: diffusion_policy/env_runner/antmaze_lowdim_runner.py

import pathlib
import logging
import numpy as np
import gym
import wandb
import math
import tqdm
import torch

from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.common.pytorch_util import dict_apply

logger = logging.getLogger(__name__)

class AntMazeLowdimRunner(BaseLowdimRunner):
    def __init__(self,
                 env_name='antmaze-medium-play-v0',
                 episodes=5,
                 max_steps=1000,
                 render_hw=(240, 360),
                 fps=30,
                 crf=22,
                 n_obs_steps=2,
                 n_action_steps=8,
                 output_dir='/home/Chen/tum-adlr-08',
                 **kwargs):
        """
        Args:
            env_name (str): Gym environment name.
            episodes (int): Total number of evaluation episodes.
            max_steps (int): Maximum steps per episode.
            render_hw (tuple): Render resolution, e.g. (240, 360).
            fps (float): Frames per second for video recording.
            crf (int): Constant rate factor for video compression.
            n_obs_steps (int): Number of observation steps per rollout.
            n_action_steps (int): Number of action steps per rollout.
            output_dir (str): Directory to save videos and logs.
        """
        super().__init__(output_dir)
        self.env_name = env_name
        self.episodes = episodes
        self.max_steps = max_steps
        self.render_hw = render_hw
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.output_dir = output_dir

    def _make_env(self):
        """
        Create a single AntMaze environment, wrap it with video recording and multi-step wrappers.
        """
        env = gym.make(self.env_name)
        # Wrap for video recording.
        env = VideoRecordingWrapper(
            env,
            video_recoder=VideoRecorder.create_h264(
                fps=self.fps,
                codec='h264',
                input_pix_fmt='rgb24',
                crf=self.crf,
                thread_type='FRAME',
                thread_count=1
            ),
            file_path=None,  # Will be set per episode
            steps_per_render=1
        )
        # Wrap with multi-step wrapper for temporal aggregation
        env = MultiStepWrapper(
            env,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            max_episode_steps=self.max_steps
        )
        return env

    def run(self, policy):
        """
        Run evaluation rollouts using the provided low-dimensional policy.
        For each episode, a fresh environment is created, a unique video file path is set,
        and the policy is run until termination or max steps are reached.
        
        Args:
            policy (BaseLowdimPolicy): The policy to evaluate. Must implement:
                - reset() method.
                - predict_action(obs_dict) returning a dict with key 'action'.
                - device attribute.
                - obs_dim attribute indicating the expected observation feature dimension.
                
        Returns:
            dict: A dictionary containing the recorded videos (wrapped as wandb.Video).
        """
        device = policy.device
        video_paths = []

        for ep in range(self.episodes):
            env = self._make_env()
            # Set a unique video file path for this episode
            video_file = pathlib.Path(self.output_dir).joinpath(
                'media', f"antmaze_ep{ep}_{wandb.util.generate_id()}.mp4"
            )
            video_file.parent.mkdir(parents=True, exist_ok=True)
            env.video_recoder.file_path = str(video_file)

            obs = env.reset()
            policy.reset()
            done = False
            step = 0
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"AntMaze Eval Ep {ep+1}/{self.episodes}", leave=False)

            n_obs_steps = self.n_obs_steps
            obs_history = [obs for _ in range(n_obs_steps)]
            
            while not done and step < self.max_steps:
                # Prepare the observation history as a tensor of shape [1, n_obs_steps, obs_dim]
                obs_array = np.array(obs_history)  # shape: (n_obs_steps, state_dim)

                desired_dim = policy.obs_dim
                current_dim = obs_array.shape[-1]
                if current_dim < desired_dim:
                    pad_size = desired_dim - current_dim
                    pad = np.zeros((n_obs_steps, pad_size), dtype=np.float32)
                    obs_array = np.concatenate([obs_array, pad], axis=-1)
                elif current_dim > desired_dim:
                    obs_array = obs_array[:, :desired_dim]
                
                # Convert to tensor and add batch dim
                obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0)  # shape: (1, n_obs_steps, obs_dim)
                obs_tensor = obs_tensor.to(device)
                obs_dict = {"obs": obs_tensor}

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # Convert back to numpy
                np_action_dict = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())
                action = np_action_dict['action']

                obs, reward, done, info = env.step(action)
                done = np.all(done)
                step += 1
                pbar.update(action.shape[1])
            pbar.close()
            env.close()
            video_paths.append(str(video_file))
            logger.info(f"Episode {ep} video saved at: {video_file}")

        wandb_videos = [wandb.Video(vp) for vp in video_paths if vp is not None]
        return {'test_sim_videos': wandb_videos}

