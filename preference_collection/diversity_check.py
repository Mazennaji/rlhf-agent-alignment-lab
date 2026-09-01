import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import numpy as np
from stable_baselines3 import PPO

from envs.base_env import make_env
from configs.config import CFG
from preference_collection.rollout import generate_pair


def action_overlap(actions_a, actions_b):
    min_len = min(len(actions_a), len(actions_b))
    if min_len == 0:
        return 1.0
    matches = np.sum(actions_a[:min_len].flatten() == actions_b[:min_len].flatten())
    return matches / min_len


def pair_diversity(pair):
    overlap = action_overlap(pair["traj_a"]["actions"], pair["traj_b"]["actions"])
    return 1.0 - overlap


def analyze_dataset(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    diversities = [pair_diversity(p) for p in data]

    print(f"Total pairs: {len(data)}")
    print(f"Diversity — mean={np.mean(diversities):.3f}, min={np.min(diversities):.3f}, max={np.max(diversities):.3f}")

    low_diversity = [d for d in diversities if d < CFG.min_pair_diversity]
    print(f"Pairs below diversity threshold ({CFG.min_pair_diversity}): {len(low_diversity)} "
          f"({len(low_diversity) / len(data):.1%})")

    return data, diversities


def filter_dataset(pkl_path: str, out_path: str):
    data, diversities = analyze_dataset(pkl_path)
    kept = [pair for pair, d in zip(data, diversities) if d >= CFG.min_pair_diversity]

    with open(out_path, "wb") as f:
        pickle.dump(kept, f)

    print(f"\n✅ Kept {len(kept)}/{len(data)} pairs above diversity threshold.")
    print(f"✅ Saved to {out_path}")
    return kept


def resample_low_diversity(pkl_path: str, out_path: str, max_attempts: int = 5):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    env = make_env(CFG.env_id, CFG.seed + 400)
    model = PPO.load("base_agent/checkpoints/base_ppo_final")

    resampled_count = 0
    for i, pair in enumerate(data):
        if pair_diversity(pair) >= CFG.min_pair_diversity:
            continue

        for attempt in range(max_attempts):
            traj_a, traj_b = generate_pair(model, env, CFG.trajectory_length)
            candidate = {
                "traj_a": traj_a,
                "traj_b": traj_b,
                "preferred": 1 if traj_a["total_return"] >= traj_b["total_return"] else 0,
            }
            if pair_diversity(candidate) >= CFG.min_pair_diversity:
                data[i] = candidate
                resampled_count += 1
                break

    env.close()

    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print(f"✅ Resampled {resampled_count} low-diversity pairs.")
    print(f"✅ Saved to {out_path}")
    return data


if __name__ == "__main__":
    pkl_path = "preference_collection/data/preference_pairs.pkl"
    out_path = "preference_collection/data/preference_pairs_diverse.pkl"
    resample_low_diversity(pkl_path, out_path)