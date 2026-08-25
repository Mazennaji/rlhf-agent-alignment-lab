import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from envs.base_env import make_env
from configs.config import CFG
from reward_model.model import RewardModel
from reward_model.dataset import PreferenceDataset, collate_fn


def bradley_terry_loss(score_a: torch.Tensor, score_b: torch.Tensor, preferred: torch.Tensor):
    logits = score_a - score_b
    return nn.functional.binary_cross_entropy_with_logits(logits, preferred)


def train_reward_model():
    env = make_env(CFG.env_id, CFG.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    dataset = PreferenceDataset(
        "preference_collection/data/preference_pairs.pkl", action_dim
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    model = RewardModel(obs_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.reward_model_lr)

    for epoch in range(CFG.reward_model_epochs):
        total_loss = 0.0
        correct = 0

        for batch in loader:
            obs_a, act_a, obs_b, act_b, preferred = batch[0]

            score_a = model.score_trajectory(obs_a, act_a)
            score_b = model.score_trajectory(obs_b, act_b)

            loss = bradley_terry_loss(score_a.unsqueeze(0), score_b.unsqueeze(0), preferred.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted_pref = 1 if score_a.item() >= score_b.item() else 0
            correct += int(predicted_pref == int(preferred.item()))

        avg_loss = total_loss / len(dataset)
        accuracy = correct / len(dataset)
        print(f"Epoch {epoch+1}/{CFG.reward_model_epochs} — loss: {avg_loss:.4f}, preference accuracy: {accuracy:.2%}")

    os.makedirs("reward_model/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "reward_model/checkpoints/reward_model.pt")
    print("✅ Reward model saved to reward_model/checkpoints/reward_model.pt")

    return model


if __name__ == "__main__":
    train_reward_model()