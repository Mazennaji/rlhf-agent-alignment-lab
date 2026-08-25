import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
from stable_baselines3 import PPO
from envs.base_env import make_env
from configs.config import CFG
from preference_collection.rollout import generate_pair


def simulated_oracle_label(traj_a, traj_b):
    return 1 if traj_a["total_return"] >= traj_b["total_return"] else 0


def build_preference_dataset():
    env = make_env(CFG.env_id, CFG.seed + 100)
    model = PPO.load("base_agent/checkpoints/base_ppo_final")

    dataset = []
    for i in range(CFG.num_trajectory_pairs):
        traj_a, traj_b = generate_pair(model, env, CFG.trajectory_length)
        label = simulated_oracle_label(traj_a, traj_b)

        dataset.append({
            "traj_a": traj_a,
            "traj_b": traj_b,
            "preferred": label,
        })

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{CFG.num_trajectory_pairs} preference pairs")

    os.makedirs("preference_collection/data", exist_ok=True)
    out_path = "preference_collection/data/preference_pairs.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"✅ Saved {len(dataset)} preference pairs to {out_path}")
    env.close()
    return dataset


if __name__ == "__main__":
    build_preference_dataset()