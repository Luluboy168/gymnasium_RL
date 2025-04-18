import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time

from model import ActorCritic

# ---------- Load Environment and Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("Walker2d-v5", render_mode="human")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

model = ActorCritic(obs_dim, act_dim).to(device)
model.load_state_dict(torch.load("models/ppo_walker2d_model.pth", map_location=device))
model.eval()

# ---------- Run Loop ----------
obs, _ = env.reset()
total_reward = 0

while True:
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        mean, std, _ = model(obs_tensor)
    action = mean.squeeze(0).cpu().numpy()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

    if done:
        print(f"Episode finished. Total Reward: {total_reward:.2f}")
        total_reward = 0
        break

env.close()
