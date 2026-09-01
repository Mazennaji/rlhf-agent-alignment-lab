import torch
from reward_model.model import RewardModel


class RewardModelEnsemble:
    def __init__(self, models):
        self.models = models

    def eval(self):
        for m in self.models:
            m.eval()

    def __call__(self, obs, action):
        scores = torch.stack([m(obs, action) for m in self.models], dim=0)
        return scores.mean(dim=0)

    def score_trajectory(self, obs, action):
        scores = torch.stack([m.score_trajectory(obs, action) for m in self.models], dim=0)
        return scores.mean()

    def score_uncertainty(self, obs, action):
        scores = torch.stack([m(obs, action) for m in self.models], dim=0)
        return scores.std(dim=0)

    @staticmethod
    def load(obs_dim, action_dim, n_models, checkpoint_dir="reward_model/checkpoints"):
        models = []
        for i in range(n_models):
            m = RewardModel(obs_dim, action_dim)
            m.load_state_dict(torch.load(f"{checkpoint_dir}/reward_model_{i}.pt"))
            m.eval()
            models.append(m)
        return RewardModelEnsemble(models)