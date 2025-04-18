# model.py

import torch
import torch.nn as nn

# ---------- Actor-Critic Network ----------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        shared_out = self.shared(x)
        mean = self.actor(shared_out)
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(shared_out).squeeze(-1)
        return mean, std, value