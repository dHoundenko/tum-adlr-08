import os
import torch
import hydra
from omegaconf import OmegaConf
import pathlib
import d4rl

# Import your workspace class
from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import TrainDiffusionUnetLowdimWorkspace

@hydra.main(config_path="diffusion_policy/config", config_name="antmaze_medium_play")
def generate_video(cfg):
    # Make sure the working directory is set to the project root if required.
    ROOT_DIR = pathlib.Path(__file__).parent.parent
    os.chdir(str(ROOT_DIR))

    # Instantiate the workspace using the config
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    
    # Path to your checkpoint
    checkpoint_path = "/home/Chen/tum-adlr-08/data/outputs/2025.01.31/08.21.33_train_diffusion_unet_lowdim_antmaze_medium_play/checkpoints/epoch=0050-train_loss=0.252.ckpt"
    
    # Load the checkpoint (this should update the model weights)
    workspace.load_checkpoint(checkpoint_path)
    
    # Select the policy model (use ema_model if available)
    policy = workspace.ema_model if workspace.ema_model is not None else workspace.model
    policy.eval()
    
    # Instantiate the environment runner from your config.
    # Note: if your workspace already set up an env_runner, you could reuse it.
    env_runner = hydra.utils.instantiate(cfg.task.env_runner)
    
    # Run the policy in the environment to generate a rollout video.
    # The run() method is expected to both execute the rollout and save the video.
    video_log = env_runner.run(policy)
    
    # video_log might be a dict containing the video path or other metrics.
    print("Rollout video generated:", video_log)

if __name__ == "__main__":
    generate_video()
