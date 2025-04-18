import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
import time

from model import ActorCritic

# Use a non-interactive backend so plots don't show automatically
plt.switch_backend('Agg')

# ---------- Environment Setup ----------

num_envs = 32
# Create a vectorized environment using gym.make_vec (without any wrappers)
envs = gym.make_vec("Walker2d-v5", num_envs=num_envs, vectorization_mode="sync")

obs_space = envs.single_observation_space  # e.g., Box(-inf, inf, (17,), float64)
act_space = envs.single_action_space         # e.g., Box(-1, 1, (6,), float32)

obs_dim = obs_space.shape[0]  # 17
act_dim = act_space.shape[0]  # 6

# We'll manually track cumulative reward per sub-environment
episode_rewards = np.zeros(num_envs)  # one per sub-environment
all_episode_rewards = []  # to store completed episode rewards
episode_count = 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCritic(obs_dim, act_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# ---------- Hyperparameters ----------
total_updates    = 1000       # total training updates
steps_per_update = 2048       # total timesteps per update (across all envs)
ppo_epochs       = 10         # PPO epochs per update
mini_batch_size  = 128        # mini-batch size for PPO updates
gamma            = 0.99       # discount factor
gae_lambda       = 0.95       # GAE lambda
clip_epsilon     = 0.1       # PPO clip parameter
value_loss_coef  = 0.25        # weight for the value loss
entropy_coef     = 0.01       # weight for the entropy bonus

# Storage for logging and plotting
loss_history      = []        # average loss per update
plot_updates      = []        # x-axis: episode count marker for plots
avg_reward_history= []        # average reward every 50 episodes
next_plot_episode = 50        # threshold for plot update

# ---------- Utility: Generalized Advantage Estimation (GAE) ----------
def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    lastgaelam = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[-1]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t+1]
            next_val = values[t+1]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        lastgaelam = delta + gamma * lam * next_non_terminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns

# ---------- Main Training Loop ----------
obs, _ = envs.reset()
for update in range(total_updates):
    start_time = time.time()
    # Buffers for rollout
    obs_buffer      = []
    actions_buffer  = []
    logprobs_buffer = []
    rewards_buffer  = []
    dones_buffer    = []
    values_buffer   = []
    
    # Collect rollout of fixed steps_per_update
    for step in range(steps_per_update):
        obs_tensor = torch.FloatTensor(obs).to(device)
        with torch.no_grad():
            mean, std, value = model(obs_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(axis=-1)
        
        # Store step data
        obs_buffer.append(obs)
        actions_buffer.append(action.cpu().numpy())
        logprobs_buffer.append(logprob.cpu().numpy())
        values_buffer.append(value.cpu().numpy())
        
        # Step the environments
        actions_np = action.cpu().numpy()
        next_obs, rewards, terminations, truncations, infos = envs.step(actions_np)
        rewards_buffer.append(rewards)
        dones_buffer.append(terminations | truncations)
        
        # Update per-environment cumulative rewards
        episode_rewards += rewards
        # If an environment is done or truncated, record its episodic reward and reset that counter.
        for i in range(num_envs):
            if terminations[i] or truncations[i]:
                all_episode_rewards.append(episode_rewards[i])
                episode_count += 1
                episode_rewards[i] = 0  # reset the cumulative reward
        
        obs = next_obs

    # Convert buffers to numpy arrays (shape: [steps_per_update, num_envs, ...])
    obs_buffer      = np.array(obs_buffer)
    actions_buffer  = np.array(actions_buffer)
    logprobs_buffer = np.array(logprobs_buffer)
    rewards_buffer  = np.array(rewards_buffer)
    dones_buffer    = np.array(dones_buffer, dtype=np.float32)
    values_buffer   = np.array(values_buffer)

    # Bootstrap: get value estimates for the last observations
    obs_tensor = torch.FloatTensor(obs).to(device)
    with torch.no_grad():
        _, _, next_values = model(obs_tensor)
    next_values = next_values.cpu().numpy()  # shape: (num_envs,)

    # Compute advantages and returns for each sub-environment separately.
    advantages_buffer = np.zeros_like(rewards_buffer)
    returns_buffer    = np.zeros_like(rewards_buffer)
    for env_idx in range(num_envs):
        adv, ret = compute_gae(
            rewards=rewards_buffer[:, env_idx],
            values=values_buffer[:, env_idx],
            dones=dones_buffer[:, env_idx],
            next_value=next_values[env_idx],
            gamma=gamma,
            lam=gae_lambda
        )
        advantages_buffer[:, env_idx] = adv
        returns_buffer[:, env_idx] = ret

    # Flatten the trajectory: shape becomes (steps_per_update * num_envs, ...)
    batch_obs       = torch.FloatTensor(obs_buffer.reshape(-1, obs_dim)).to(device)
    batch_actions   = torch.FloatTensor(actions_buffer.reshape(-1, act_dim)).to(device)
    batch_logprobs  = torch.FloatTensor(logprobs_buffer.reshape(-1)).to(device)
    batch_returns   = torch.FloatTensor(returns_buffer.reshape(-1)).to(device)
    batch_advantages= torch.FloatTensor(advantages_buffer.reshape(-1)).to(device)
    # Normalize advantages
    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
    batch_advantages = torch.clamp(batch_advantages, -10, 10)

    # ---------- PPO Policy Update ----------
    total_loss = 0
    batch_size = batch_obs.shape[0]
    indices = np.arange(batch_size)
    
    for epoch in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb_idx = indices[start:end]
            
            minibatch_obs       = batch_obs[mb_idx]
            minibatch_actions   = batch_actions[mb_idx]
            minibatch_logprobs  = batch_logprobs[mb_idx]
            minibatch_returns   = batch_returns[mb_idx]
            minibatch_advantages= batch_advantages[mb_idx]
            
            mean, std, values = model(minibatch_obs)
            dist = torch.distributions.Normal(mean, std)
            new_logprobs = dist.log_prob(minibatch_actions).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1).mean()
            
            ratio = torch.exp(new_logprobs - minibatch_logprobs)
            surr1 = ratio * minibatch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * minibatch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((minibatch_returns - values) ** 2).mean()
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    avg_loss = total_loss / (ppo_epochs * (batch_size // mini_batch_size))
    loss_history.append(avg_loss)
    
    # Print training information for every update
    recent_avg_reward = (np.mean(all_episode_rewards[-10:])
                         if len(all_episode_rewards) >= 10 else
                         np.mean(all_episode_rewards) if all_episode_rewards else 0)
    time_elapsed = time.time() - start_time
    print(f"Update {update+1:4d} | Avg Loss: {avg_loss:6.3f} | Episodes: {episode_count:4d} | Recent Avg Reward: {recent_avg_reward:6.3f} | Time: {time_elapsed:.2f}")
    
    # Save model every 50 updates and save the training plot
    if (update + 1) % 50 == 0:
        model_path = f"models/ppo_walker2d_update_{update+1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at update {update+1} as '{model_path}'")
        
        # Save a plot of average reward and loss.
        # Plot rewards: individual and rolling mean
        plt.figure(figsize=(12, 6))
        plt.plot(all_episode_rewards, label="Episode Reward")

        # Moving average over last 500
        if len(all_episode_rewards) >= 500:
            rolling = pd.Series(all_episode_rewards).rolling(window=500).mean()
            plt.plot(rolling, label="Moving Average (500)", linewidth=2)

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Reward Progress")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("figures/episode_rewards.png")
        plt.close()

        # Plot loss in a separate figure
        plt.figure(figsize=(10, 5))
        updates_axis = np.arange(1, len(loss_history) + 1)
        plt.plot(updates_axis, loss_history, marker='o')
        plt.xlabel("Update")
        plt.ylabel("Average Loss")
        plt.title("Loss Progress")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("figures/loss_plot.png")
        plt.close()

# ---------- Final Save ----------
torch.save(model.state_dict(), "models/ppo_walker2d_model.pth")
print("Final model saved as 'ppo_walker2d_model.pth'")
