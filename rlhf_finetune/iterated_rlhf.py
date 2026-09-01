import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs.base_env import make_env
from configs.config import CFG
from preference_collection.rollout import generate_pair
from reward_model.model import RewardModel
from reward_model.dataset import PreferenceDataset, collate_fn, train_val_split
from reward_model.ensemble import RewardModelEnsemble
from rlhf_finetune.reward_wrapper import RLHFRewardWrapper, ModelRef
from rlhf_finetune.kl_schedule import linear_kl_schedule
from torch.utils.data import DataLoader


def generate_pairs_from_model(model_path, out_path, n_pairs, seed_offset):
    env = make_env(CFG.env_id, CFG.seed + seed_offset)
    model = PPO.load(model_path)

    dataset = []
    for _ in range(n_pairs):
        traj_a, traj_b = generate_pair(model, env, CFG.trajectory_length)
        preferred = 1 if traj_a["total_return"] >= traj_b["total_return"] else 0
        dataset.append({"traj_a": traj_a, "traj_b": traj_b, "preferred": preferred})

    env.close()

    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)

    return dataset


def merge_pickles(paths, out_path):
    combined = []
    for p in paths:
        with open(p, "rb") as f:
            combined.extend(pickle.load(f))
    with open(out_path, "wb") as f:
        pickle.dump(combined, f)
    return combined


def bradley_terry_loss(score_a, score_b, preferred):
    logits = score_a - score_b
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, preferred)


def train_one_model(model_idx, obs_dim, action_dim, pkl_path):
    train_idx, val_idx = train_val_split(pkl_path, CFG.reward_model_val_split, seed=CFG.seed + model_idx)

    train_dataset = PreferenceDataset(pkl_path, action_dim, indices=train_idx)
    val_dataset = PreferenceDataset(pkl_path, action_dim, indices=val_idx)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    torch.manual_seed(CFG.seed + model_idx)
    model = RewardModel(obs_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.reward_model_lr)

    for epoch in range(CFG.reward_model_epochs):
        for batch in train_loader:
            obs_a, act_a, obs_b, act_b, preferred = batch[0]
            score_a = model.score_trajectory(obs_a, act_a)
            score_b = model.score_trajectory(obs_b, act_b)
            loss = bradley_terry_loss(score_a.unsqueeze(0), score_b.unsqueeze(0), preferred.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    correct, n = 0, 0
    for batch in val_loader:
        obs_a, act_a, obs_b, act_b, preferred = batch[0]
        with torch.no_grad():
            score_a = model.score_trajectory(obs_a, act_a)
            score_b = model.score_trajectory(obs_b, act_b)
        predicted = 1 if score_a.item() >= score_b.item() else 0
        correct += int(predicted == int(preferred.item()))
        n += 1
    val_acc = correct / n if n > 0 else 0.0

    return model, val_acc


def finetune_round(base_checkpoint, save_path, reward_model, timesteps, seed_offset):
    base_model = PPO.load(base_checkpoint)
    model_ref = ModelRef()
    kl_schedule = linear_kl_schedule(start=0.3, end=0.02, total_steps=timesteps)

    raw_env = make_env(CFG.env_id, CFG.seed + seed_offset)
    action_dim = raw_env.action_space.n
    wrapped_env = RLHFRewardWrapper(
        raw_env, reward_model, base_model, model_ref, action_dim, kl_coef_fn=kl_schedule
    )
    wrapped_env = Monitor(wrapped_env)

    finetuned_model = PPO.load(base_checkpoint, env=wrapped_env)
    model_ref.model = finetuned_model
    finetuned_model.learn(total_timesteps=timesteps, progress_bar=True)

    finetuned_model.save(save_path)
    wrapped_env.close()
    return finetuned_model


def run_iterated_rlhf(n_rounds=3, pairs_per_round=200):
    os.makedirs("rlhf_finetune/checkpoints/rounds", exist_ok=True)
    os.makedirs("preference_collection/data/rounds", exist_ok=True)

    current_policy_path = "base_agent/checkpoints/base_ppo_final"
    pair_files = ["preference_collection/data/preference_pairs.pkl"]

    for round_idx in range(n_rounds):
        print(f"\n===== ROUND {round_idx + 1}/{n_rounds} =====")

        round_pairs_path = f"preference_collection/data/rounds/round_{round_idx}.pkl"
        generate_pairs_from_model(
            current_policy_path, round_pairs_path, pairs_per_round, seed_offset=1000 + round_idx
        )
        pair_files.append(round_pairs_path)

        combined_path = f"preference_collection/data/rounds/combined_round_{round_idx}.pkl"
        merge_pickles(pair_files, combined_path)

        probe_env = make_env(CFG.env_id, CFG.seed)
        obs_dim = probe_env.observation_space.shape[0]
        action_dim = probe_env.action_space.n
        probe_env.close()

        models = []
        val_accs = []
        for i in range(CFG.reward_model_ensemble_size):
            model, val_acc = train_one_model(i, obs_dim, action_dim, combined_path)
            ckpt_path = f"reward_model/checkpoints/round_{round_idx}_model_{i}.pt"
            torch.save(model.state_dict(), ckpt_path)
            models.append(model)
            val_accs.append(val_acc)

        print(f"Round {round_idx + 1} reward model val accuracy — mean={sum(val_accs)/len(val_accs):.2%}")

        reward_model = RewardModelEnsemble(models)
        reward_model.eval()

        round_checkpoint_path = f"rlhf_finetune/checkpoints/rounds/round_{round_idx}"
        finetune_round(
            current_policy_path,
            round_checkpoint_path,
            reward_model,
            CFG.finetune_timesteps,
            seed_offset=2000 + round_idx,
        )

        current_policy_path = round_checkpoint_path

    print(f"\n✅ Iterated RLHF complete. Final policy at {current_policy_path}.zip")
    return current_policy_path


if __name__ == "__main__":
    run_iterated_rlhf(n_rounds=3, pairs_per_round=200)