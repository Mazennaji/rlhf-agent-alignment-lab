import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from stable_baselines3 import PPO

from envs.base_env import make_env
from configs.config import CFG
from reward_model.ensemble import RewardModelEnsemble
from preference_collection.rollout import collect_trajectory


def score_trajectory_with_ensemble(reward_model, action_dim, traj):
    obs = torch.tensor(traj["observations"], dtype=torch.float32)
    actions = traj["actions"].astype(np.int64).reshape(-1)
    actions_onehot = torch.nn.functional.one_hot(torch.tensor(actions), num_classes=action_dim).float()
    with torch.no_grad():
        return reward_model.score_trajectory(obs, actions_onehot).item()


def best_of_n_episode(model, env, reward_model, action_dim, n, max_steps=500):
    candidates = [collect_trajectory(model, env, max_steps, deterministic=False) for _ in range(n)]
    scores = [score_trajectory_with_ensemble(reward_model, action_dim, c) for c in candidates]
    best_idx = int(np.argmax(scores))
    best = candidates[best_idx]
    return best["total_return"], scores[best_idx]


def evaluate_best_of_n(n=8, n_episodes=30):
    env = make_env(CFG.env_id, CFG.seed + 700)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    reward_model = RewardModelEnsemble.load(obs_dim, action_dim, CFG.reward_model_ensemble_size)
    base_model = PPO.load("base_agent/checkpoints/base_ppo_final")

    env_returns, learned_scores = [], []
    for _ in range(n_episodes):
        env_ret, learned_score = best_of_n_episode(base_model, env, reward_model, action_dim, n)
        env_returns.append(env_ret)
        learned_scores.append(learned_score)

    env.close()

    print(f"Best-of-{n} baseline over {n_episodes} episodes")
    print(f"Env reward:   mean={np.mean(env_returns):.2f}  std={np.std(env_returns):.2f}")
    print(f"Reward model: mean={np.mean(learned_scores):.2f}  std={np.std(learned_scores):.2f}")

    return env_returns, learned_scores


if __name__ == "__main__":
    evaluate_best_of_n(n=8, n_episodes=30)