import gymnasium as gym
import tensorflow as tf
import numpy as np
import time
import ale_py
from utils import preprocess_observation

def combine_streams(inputs):
    value, advantage = inputs
    return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

# Load the trained model
loaded_model = tf.keras.models.load_model("models/dqn_assault_model.keras",
                                          custom_objects={"combine_streams": combine_streams})

# Create the environment
gym.register_envs(ale_py)
env = gym.make("ALE/Assault-v5", render_mode='human')
obs, _ = env.reset()

# Initialize state
state = preprocess_observation(obs)
state = np.repeat(state, 4, axis=-1)

done = False
episode_reward = 0

while not done:
    # Expand the state dimensions to simulate a batch of 1.
    q_values = loaded_model(np.expand_dims(state, axis=0))
    action = np.argmax(q_values.numpy()[0])
    
    # Execute the action in the environment.
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    episode_reward += reward
    # Process the observation and update the frame stack.
    next_frame = preprocess_observation(obs)
    # Update state: remove the oldest frame and add the new one.
    next_state = np.concatenate([state[..., 1:], next_frame], axis=-1)
    state = next_state

env.close()
print(f"Final Score: {episode_reward}")
