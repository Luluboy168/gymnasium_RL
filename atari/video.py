import os
import gymnasium as gym
import tensorflow as tf
import numpy as np
import imageio
import ale_py
import cv2
from utils import preprocess_observation

# -------------------------------
# Parameters and Model Loading
# -------------------------------

def combine_streams(inputs):
    value, advantage = inputs
    return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

model_path = os.path.join("models", "dqn_assault_model.keras")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the trained model
model = tf.keras.models.load_model(model_path, custom_objects={"combine_streams": combine_streams})
print("Model loaded from:", model_path)

# -------------------------------
# Environment Setup
# -------------------------------
gym.register_envs(ale_py)
env = gym.make("ALE/Assault-v5", render_mode="rgb_array")

# -------------------------------
# Initialize the Episode
# -------------------------------
obs, info = env.reset()
obs_proc = preprocess_observation(obs)

if obs_proc.ndim == 3 and obs_proc.shape[-1] == 1:
    obs_proc = np.squeeze(obs_proc, axis=-1)
# Create the initial state
state = np.stack([obs_proc] * 4, axis=-1)

done = False
video_frames = []


while not done:
    frame = env.render() 
    video_frames.append(frame)

    # Add a batch dimension and get Q-values from the model.
    state_input = np.expand_dims(state, axis=0)
    q_values = model(state_input)
    action = np.argmax(q_values.numpy())

    # Step the environment with the chosen action.
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Process the new observation.
    next_obs_proc = preprocess_observation(next_obs)

    if next_obs_proc.ndim == 3 and next_obs_proc.shape[-1] == 1:
        next_obs_proc = np.squeeze(next_obs_proc, axis=-1)  # Now (84,84)
    # Update state by removing the oldest frame and appending the new one.
    state = np.concatenate([state[..., 1:], np.expand_dims(next_obs_proc, axis=-1)], axis=-1)

env.close()

target_width = 1600
target_height = 2100

resized_frames = [
    cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    for frame in video_frames
]


os.makedirs("videos", exist_ok=True)
video_path = "videos/video2.mp4"
fps = 30 
imageio.mimsave(video_path, resized_frames, fps=fps)
print("Video saved to:", video_path)
