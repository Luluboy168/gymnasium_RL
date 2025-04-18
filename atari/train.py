import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda, Dropout
import matplotlib.pyplot as plt
import random
import ale_py
import os
import time
from collections import deque
from utils import preprocess_observation

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Error setting memory growth:", e)

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Build vectorized environment
num_envs = 32
gym.register_envs(ale_py)
envs = gym.make_vec("ALE/Assault-v5", num_envs=num_envs, vectorization_mode="async")

# Hyperparameters
input_shape = (84, 84, 4)  # grayscale, stacked
num_actions = envs.single_action_space.n
gamma = 0.99
batch_size = 1024
replay_capacity = 350000
min_replay_size = 10000
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.999995
target_update_freq = 1000
update_every = 4

# dueling DQN networks
def build_q_network(input_shape, num_actions):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, 8, strides=4, activation='relu')(inputs)
    x = Conv2D(128, 4, strides=2, activation='relu')(x)
    x = Conv2D(128, 3, strides=1, activation='relu')(x)
    x = Flatten()(x)

    # Value stream
    value = Dense(1024, activation='relu')(x)
    value = Dropout(0.1)(value)
    value = Dense(1)(value)

    # Advantage stream
    advantage = Dense(1024, activation='relu')(x)
    advantage = Dropout(0.1)(advantage)
    advantage = Dense(num_actions)(advantage)

    # Combine streams with Lambda to safely handle symbolic tensors
    def combine_streams(inputs):
        value, advantage = inputs
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

    q_values = Lambda(combine_streams, output_shape=(int(num_actions),))([value, advantage])

    model = Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.00025), loss='huber')
    return model

primary_network = build_q_network(input_shape, num_actions)
target_network = build_q_network(input_shape, num_actions)
target_network.set_weights(primary_network.get_weights())

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add_batch(self, states, actions, rewards, next_states, dones):
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.buffer.append((s, a, np.float32(r), ns, np.float32(d)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(replay_capacity)

@tf.function
def train_step(states, actions, rewards, next_states, dones):
    rewards = tf.cast(rewards, tf.float32)
    dones = tf.cast(dones, tf.float32)
    next_q_values = target_network(next_states)
    max_next_q = tf.reduce_max(next_q_values, axis=1)
    target_q = rewards + (1. - dones) * gamma * max_next_q
    with tf.GradientTape() as tape:
        q_values = primary_network(states)
        indices = tf.stack([tf.range(tf.shape(actions)[0]), tf.cast(actions, tf.int32)], axis=1)
        q_selected = tf.gather_nd(q_values, indices)
        loss = tf.keras.losses.Huber()(target_q, q_selected)
    grads = tape.gradient(loss, primary_network.trainable_variables)
    primary_network.optimizer.apply_gradients(zip(grads, primary_network.trainable_variables))
    return loss

# Training loop
total_steps = 0
num_episodes = 5000
loss_history = []
reward_history = []

for episode in range(num_episodes):
    start_time = time.time()
    obs, _ = envs.reset()
    obs = preprocess_observation(obs)
    frames = np.repeat(obs, 4, axis=-1)  # (8, 84, 84, 4)

    done_flags = np.zeros(num_envs, dtype=bool)
    episode_rewards = np.zeros(num_envs)
    episode_losses = []

    # re-explore
    if((episode) % 200 == 0):
        epsilon = max(1.0 - (episode / num_episodes) * 2, epsilon_min)

    while not np.all(done_flags):
        total_steps += 1

        # ε-greedy action selection
        q_values = primary_network(frames)
        greedy_actions = np.argmax(q_values.numpy(), axis=1)
        random_actions = np.random.randint(num_actions, size=num_envs)
        actions = np.where(np.random.rand(num_envs) < epsilon, random_actions, greedy_actions)

        next_obs, rewards, terminations, truncations, _ = envs.step(actions)
        dones = np.logical_or(terminations, truncations)
        next_obs = preprocess_observation(next_obs)
        next_frames = np.concatenate([frames[..., 1:], next_obs], axis=-1)  # (8, 84, 84, 4)

        replay_buffer.add_batch(frames, actions, rewards, next_frames, dones)
        frames = next_frames
        done_flags = np.logical_or(done_flags, dones)
        episode_rewards += rewards

        if len(replay_buffer) >= min_replay_size and total_steps % update_every == 0:
            states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)
            loss = train_step(states_b, actions_b, rewards_b, next_states_b, dones_b)
            episode_losses.append(loss.numpy())

            if total_steps % target_update_freq == 0:
                target_network.set_weights(primary_network.get_weights())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    avg_loss = np.mean(episode_losses) if episode_losses else 0
    loss_history.append(avg_loss)
    reward_history.append(np.mean(episode_rewards))
    elapsed = time.time() - start_time
    print(f"Episode {episode + 1}, Reward: {np.mean(episode_rewards):.2f}, Epsilon: {epsilon:.3f}, Total steps: {total_steps}, Time elapsed: {elapsed}")

    if (episode + 1) % 50 == 0:
        model_path = f"models/dqn_assault_model_{episode + 1}.keras"
        primary_network.save(model_path)
        print(f"Saved model to {model_path}")
        # Plotting
        plt.figure(figsize=(10, 4))
        plt.plot(loss_history, label="Avg Loss per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Episodes")
        plt.legend()
        plt.savefig("figures/loss_over_episodes.png")

        plt.figure(figsize=(10, 4))
        plt.plot(reward_history, label="Average Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Reward Over Time")
        plt.legend()
        plt.savefig("figures/reward_over_episodes.png")

envs.close()

