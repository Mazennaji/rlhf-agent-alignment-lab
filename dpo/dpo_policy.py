import torch
import torch.nn as nn


class DPOPolicy(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def log_prob_of_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs)
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1).long()).squeeze(1)
        return action_log_probs

    def trajectory_log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.log_prob_of_actions(obs, actions).sum()