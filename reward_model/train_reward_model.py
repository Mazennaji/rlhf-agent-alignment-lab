import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from envs.base_env import make_env
from configs.config import CFG
from reward_model.model import RewardModel
from reward_model.dataset import PreferenceDataset, collate_fn, train_val_split


def bradley_terry_loss(score_a, score_b, preferred):
    logits = score_a - score_b
    return nn.functional.binary_cross_entropy_with_logits(logits, preferred)


def run_split(model, loader, optimizer=None):
    is_train = optimizer is not None
    total_loss, correct, n = 0.0, 0, 0

    for batch in loader:
        obs_a, act_a, obs_b, act_b, preferred = batch[0]

        if is_train:
            score_a = model.score_trajectory(obs_a, act_a)
            score_b = model.score_trajectory(obs_b, act_b)
        else:
            with torch.no_grad():
                score_a = model.score_trajectory(obs_a, act_a)
                score_b = model.score_trajectory(obs_b, act_b)

        loss = bradley_terry_loss(score_a.unsqueeze(0), score_b.unsqueeze(0), preferred.unsqueeze(0))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        predicted_pref = 1 if score_a.item() >= score_b.item() else 0
        correct += int(predicted_pref == int(preferred.item()))
        n += 1

    return total_loss / n, correct / n


def train_one_model(model_idx, obs_dim, action_dim, pkl_path):
    train_idx, val_idx = train_val_split(pkl_path, CFG.reward_model_val_split, seed=CFG.seed + model_idx)

    train_dataset = PreferenceDataset(pkl_path, action_dim, indices=train_idx)
    val_dataset = PreferenceDataset(pkl_path, action_dim, indices=val_idx)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    torch.manual_seed(CFG.seed + model_idx)
    model = RewardModel(obs_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.reward_model_lr)

    history = []
    for epoch in range(CFG.reward_model_epochs):
        train_loss, train_acc = run_split(model, train_loader, optimizer)
        val_loss, val_acc = run_split(model, val_loader, optimizer=None)

        print(f"[model {model_idx}] epoch {epoch+1}/{CFG.reward_model_epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

    return model, history


def train_reward_model_ensemble():
    env = make_env(CFG.env_id, CFG.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    pkl_path = "preference_collection/data/preference_pairs.pkl"
    os.makedirs("reward_model/checkpoints", exist_ok=True)
    os.makedirs("reward_model/logs", exist_ok=True)

    all_history = {}
    for i in range(CFG.reward_model_ensemble_size):
        model, history = train_one_model(i, obs_dim, action_dim, pkl_path)
        torch.save(model.state_dict(), f"reward_model/checkpoints/reward_model_{i}.pt")
        all_history[f"model_{i}"] = history

    with open("reward_model/logs/calibration_history.json", "w") as f:
        json.dump(all_history, f, indent=2)

    final_val_accs = [all_history[f"model_{i}"][-1]["val_acc"] for i in range(CFG.reward_model_ensemble_size)]
    print(f"\n✅ Trained {CFG.reward_model_ensemble_size} reward models.")
    print(f"Final val accuracy — mean={sum(final_val_accs)/len(final_val_accs):.2%}, "
          f"min={min(final_val_accs):.2%}, max={max(final_val_accs):.2%}")
    print("✅ Calibration history saved to reward_model/logs/calibration_history.json")


if __name__ == "__main__":
    train_reward_model_ensemble()