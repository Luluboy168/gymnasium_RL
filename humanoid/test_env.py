import gymnasium as gym
# import pybullet_envs

env = gym.make("Walker2d-v5", render_mode="human")
obs, _ = env.reset()

done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

env.close()
print("Total reward:", total_reward)
