# Gymnasium Agents with Deep Reinforcement Learning 🎮

This repository contains three different Deep Reinforcement Learning agent to play three different environments from [OpenAI Gymnasium](https://gymnasium.farama.org). The implementation supports training, playing, and generating gameplay videos using pre-trained models. 
1. [Assault](https://ale.farama.org/environments/assault/) agent trained using DQN.
2. [Walker2D](https://gymnasium.farama.org/environments/mujoco/walker2d/) agent trained using PPO.
3. [Humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/) agent trained using PPO.

## Author
111550056 陳晉祿 (luluboy)

## 🛠️ Project Structure
Here's the overall file structure of this project 
```
. 
├── atari/ 
│ ├── experiments/ # experiment results
│ ├── figures/ # Training performance plots 
│ ├── models/ # Saved models 
│ ├── videos/ # Recorded gameplay videos 
│ ├── play.py # Play with a trained game agent 
│ ├── train.py # Train game agent 
│ ├── utils.py # Observation preprocessing 
│ └── video.py # Save game agent gameplay to video 
├── humanoid/ 
│ ├── figures/
│ ├── models/ 
│ ├── videos/ 
│ ├── model.py # Actor-Critic model architecture 
│ ├── play.py # Play with PPO agent 
│ ├── test_env.py # Environment test script 
│ ├── train.py # PPO training script 
│ └── video.py # Generate PPO gameplay video 
├── walker/ 
│ ├── figures/ 
│ ├── models/ 
│ ├── videos/ 
│ ├── model.py
│ ├── play.py
│ ├── test_env.py
│ ├── train.py
│ └── video.py
├── README.md  
└── requirements.txt 
```



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

## Demo video clips
[![Watch the video](https://img.youtube.com/vi/zK9K6LRh2B8/hqdefault.jpg)](https://www.youtube.com/watch?v=zK9K6LRh2B8)
[![Watch the video](https://img.youtube.com/vi/J9E-zQH21Pc/hqdefault.jpg)](https://www.youtube.com/watch?v=J9E-zQH21Pc)
[![Watch the video](https://img.youtube.com/vi/zPHR8DiUPH0/hqdefault.jpg)](https://www.youtube.com/watch?v=zPHR8DiUPH0)

<p align="center">
  <a href="https://www.youtube.com/watch?v=zK9K6LRh2B8">
    <img src="https://img.youtube.com/vi/zK9K6LRh2B8/hqdefault.jpg" width="30%" />
  </a>
  <a href="https://www.youtube.com/watch?v=J9E-zQH21Pc">
    <img src="https://img.youtube.com/vi/J9E-zQH21Pc/hqdefault.jpg" width="30%" />
  </a>
  <a href="https://www.youtube.com/watch?v=zPHR8DiUPH0">
    <img src="https://img.youtube.com/vi/zPHR8DiUPH0/hqdefault.jpg" width="30%" />
  </a>
</p>

## References
- https://gymnasium.farama.org/  
- https://github.com/Harsha1997/DeepLearning-in-Atari-Games  
- https://ithelp.ithome.com.tw/articles/10225812  
- https://www.youtube.com/playlist?list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_  
- https://github.com/PawelMlyniec/Walker-2D  
- https://hackmd.io/@RL666/rJUDS6K05  
- https://medium.com/intro-to-artificial-intelligence/the-actor-critic-reinforcement-learning-algorithm-c8095a655c14  
- https://andy6804tw.github.io/2022/04/03/python-video-save/  

