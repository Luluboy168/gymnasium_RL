import gymnasium as gym
import mujoco
import numpy as np
import torch
import imageio
import os

from model import ActorCritic  # Make sure you have this defined in model.py

# ---------- Config ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("Humanoid-v5", render_mode=None)

height, width = 1080, 1920

# ---------- Access raw MuJoCo model and data ----------
model = env.unwrapped.model
data = env.unwrapped.data

# ---------- Patch framebuffer size for high-res rendering ----------
model.vis.global_.offwidth = width
model.vis.global_.offheight = height

renderer = mujoco.Renderer(model, height=height, width=width)
cam = mujoco.MjvCamera()
cam.azimuth = 90
cam.elevation = -20
cam.distance = 4.0
cam.lookat[:] = [0.0, 0.0, 1.0]   # Initial camera center

# ---------- Load trained PPO policy ----------
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

policy = ActorCritic(obs_dim, act_dim).to(device)
policy.load_state_dict(torch.load("models/ppo_humanoid_model.pth", map_location=device))
policy.eval()

# ---------- Run simulation and record frames ----------
obs, _ = env.reset()
total_reward = 0
frames = []

max_steps = 2000
for step in range(max_steps):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mean, std, _ = policy(obs_tensor)
    action = mean.squeeze(0).cpu().numpy()

    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

    # Update camera to follow the agent
    cam.lookat[0] = data.qpos[0]   # x
    cam.lookat[1] = data.qpos[1]   # y
    cam.lookat[2] = 0.6 

    # Render
    renderer.update_scene(data, camera=cam)
    frame = renderer.render()
    frames.append(np.copy(frame))

    if terminated or truncated:
        print(f"Episode finished early at step {step}. Total reward: {total_reward:.2f}")
        break

env.close()

# ---------- Save video ----------
os.makedirs("videos", exist_ok=True)
video_path = "videos/video.mp4"
imageio.mimsave(video_path, frames, fps=45)
print(f"Video saved to: {video_path}")
