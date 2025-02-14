import os
import torch
import gym
import numpy as np
import imageio
from omegaconf import OmegaConf
import hydra
from hydra import initialize, compose
import d4rl

# -----------------------------------------------------------------------------
# Register the "eval" resolver before loading any config.
# This enables the use of ${eval:...} interpolations in your YAML config.
# -----------------------------------------------------------------------------
OmegaConf.register_new_resolver("eval", eval, replace=True)

# -----------------------------------------------------------------------------
# Load the full AntMaze config from your YAML file.
# -----------------------------------------------------------------------------
def load_config():
    # Use a relative path (from the working directory) to your config folder.
    config_dir = "diffusion_policy/config"
    with initialize(config_path=config_dir, version_base=None):
        cfg = compose(config_name="antmaze_medium_play")
    OmegaConf.resolve(cfg)
    return cfg

# -----------------------------------------------------------------------------
# Instantiate the diffusion policy from the full config and load the checkpoint.
# -----------------------------------------------------------------------------
def load_policy_from_checkpoint(cfg, checkpoint_path: str, device: torch.device):
    # Instantiate the policy from the full config.
    policy = hydra.utils.instantiate(cfg.policy)
    policy.to(device)
    
    # Load the checkpoint.
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Extract the model state dictionary.
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "state_dicts" in ckpt:
        # Assuming that the model parameters are stored under the key "model"
        state_dict = ckpt["state_dicts"]["model"]
    else:
        state_dict = ckpt  # Fallback (if the checkpoint is a plain state dict)
    
    # Load the state dictionary into the policy.
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


# -----------------------------------------------------------------------------
# Run a rollout in the environment and record a video.
# -----------------------------------------------------------------------------
def generate_rollout_video(policy, cfg, video_filename="antmaze_rollout2.mp4", max_steps=1000, fps=30):
    # Use the environment name from the config.
    env_name = cfg.task.env_runner.env_name
    env = gym.make(env_name)
    obs = env.reset()

    # Number of observation steps is specified in the top-level config.
    n_obs_steps = cfg.n_obs_steps
    obs_history = [obs for _ in range(n_obs_steps)]
    frames = []

    step = 0
    done = False
    while not done and step < max_steps:
        # Prepare the observation history as a tensor of shape [1, n_obs_steps, obs_dim]
        obs_array = np.array(obs_history)  # shape: (n_obs_steps, obs_dim)
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        obs_tensor = obs_tensor.to(next(policy.parameters()).device)
        obs_dict = {"obs": obs_tensor}

        with torch.no_grad():
            result = policy.predict_action(obs_dict)

        # The policy output may use either "action_pred" or "action".
        if "action_pred" in result:
            action_seq = result["action_pred"]
        elif "action" in result:
            action_seq = result["action"]
        else:
            raise ValueError("Policy output did not contain a valid action key.")

        # For closed-loop control, use the first action in the predicted sequence.
        action = action_seq[0, 0].cpu().numpy()

        # Step the environment.
        obs, reward, done, info = env.step(action)

        # Update the observation history.
        obs_history.pop(0)
        obs_history.append(obs)

        # Render the current frame (requires env.render(mode="rgb_array") support)
        frame = env.render(mode="rgb_array")
        frames.append(frame)

        step += 1

    env.close()

    # Save the recorded frames as a video.
    imageio.mimwrite(video_filename, frames, fps=fps)
    print(f"Video saved to {video_filename}")

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main():
    # Specify the checkpoint path.
    checkpoint_path = (
        "/home/Chen/tum-adlr-08/data/outputs/2025.02.07/14.13.26_train_diffusion_unet_lowdim_antmaze_medium_play/checkpoints/latest.ckpt"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the full configuration from the YAML file.
    cfg = load_config()

    # Instantiate the policy and load weights from the checkpoint.
    policy = load_policy_from_checkpoint(cfg, checkpoint_path, device)

    # Use the rollout parameters from the config (or override if needed).
    max_steps = 1000
    fps = cfg.task.env_runner.fps if "fps" in cfg.task.env_runner else 30

    # Generate and save the rollout video.
    generate_rollout_video(policy, cfg, video_filename="antmaze_rollout2.mp4", max_steps=max_steps, fps=fps)

if __name__ == "__main__":
    main()
