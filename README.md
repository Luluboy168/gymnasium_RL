# Gymnasium Agents with Deep Reinforcement Learning 🎮

This repository contains three different Deep Reinforcement Learning agent to play three different environments from [OpenAI Gymnasium](https://gymnasium.farama.org). The implementation supports training, playing, and generating gameplay videos using pre-trained models. 
1. [Assault](https://ale.farama.org/environments/assault/) agent trained using DQN.
2. [Walker2D](https://gymnasium.farama.org/environments/mujoco/walker2d/) agent trained using PPO.
3. [Humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/) agent trained using PPO.

## 🛠️ Project Structure
```
. 
├── train.py # Train the agent 
├── play.py # Play with a trained agent 
├── video.py # Record video from trained agent 
├── model/ # Folder for saving trained models 
├── figures/ # Folder for saving training plots 
├── utils.py # Preprocessing functions 
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

Create and activate your virtual environment (optional but recommended), then run:

```bash
pip install -r requirements.txt
```
### 2. Choose environment
```bash
cd atari
# cd humanoid
# cd walker
```
### 3. Train the Agent
To start training the agent on an Atari game environment (e.g., ALE/Assault-v5):
```bash
python train.py
```
Training progress (e.g., rewards per episode) will be saved to figures/, and models will be saved to models/.

### 4. Play with the Trained Model
After training completes, you can run the trained agent:

```bash
python play.py
```
This will load the model from models/ and run the game with live inference.

### 5. Generate Gameplay Video
To record the gameplay into a video file:

```bash
python video.py
```
The video will be saved to the current working directory.

🧠 Features
- ✅ Convolutional Neural Network-based Q-Network

- ✅ Support for DQN and extensions (e.g., Double DQN, Dueling DQN)

- ✅ Frame preprocessing and resizing (84x84 grayscale)

- ✅ Experience replay

- ✅ GPU support with memory growth

- ✅ Vectorized environments (parallel training support)