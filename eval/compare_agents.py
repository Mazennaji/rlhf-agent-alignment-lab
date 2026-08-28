import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from envs.base_env import make_env
from configs.config import CFG
from reward_model.model import RewardModel


def load_reward_model(obs_dim: int, action_dim: int) -> RewardModel:
    model = RewardModel(obs_dim, action_dim)
    model.load_state_dict(torch.load("reward_model/checkpoints/reward_model.pt"))
    model.eval()
    return model


def run_episode(model, env, reward_model, action_dim, max_steps=500):
    obs, info = env.reset()
    env_return, learned_return = 0.0, 0.0

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_onehot = torch.nn.functional.one_hot(
            torch.tensor([int(action)]), num_classes=action_dim
        ).float()
        with torch.no_grad():
            learned_reward = reward_model(obs_t, action_onehot).item()

        env_return += reward
        learned_return += learned_reward
        obs = next_obs

        if terminated or truncated:
            break

    return env_return, learned_return


def evaluate(model, label, env, reward_model, action_dim, n_episodes=20):
    env_returns, learned_returns = [], []
    for _ in range(n_episodes):
        env_ret, learned_ret = run_episode(model, env, reward_model, action_dim)
        env_returns.append(env_ret)
        learned_returns.append(learned_ret)

    print(f"\n--- {label} ---")
    print(f"Env reward:     mean={np.mean(env_returns):.2f}  std={np.std(env_returns):.2f}")
    print(f"Reward model:   mean={np.mean(learned_returns):.2f}  std={np.std(learned_returns):.2f}")

    return env_returns, learned_returns


def compare_agents():
    env = make_env(CFG.env_id, CFG.seed + 999)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    reward_model = load_reward_model(obs_dim, action_dim)

    base_model = PPO.load("base_agent/checkpoints/base_ppo_final")
    rlhf_model = PPO.load("rlhf_finetune/checkpoints/rlhf_ppo_final")

    base_env_r, base_learned_r = evaluate(base_model, "Base PPO Agent", env, reward_model, action_dim)
    rlhf_env_r, rlhf_learned_r = evaluate(rlhf_model, "RLHF-Tuned Agent", env, reward_model, action_dim)

    env.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([base_env_r, rlhf_env_r], tick_labels=["Base PPO", "RLHF-Tuned"])
    ax.set_title("Raw Environment Reward per Episode")
    ax.set_ylabel("Episode Return")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.boxplot([base_learned_r, rlhf_learned_r], tick_labels=["Base PPO", "RLHF-Tuned"])
    ax2.set_title("Reward Model Score per Episode")
    ax2.set_ylabel("Learned Reward Score")

    os.makedirs("eval/results", exist_ok=True)
    fig.savefig("eval/results/env_reward_comparison.png")
    fig2.savefig("eval/results/reward_model_comparison.png")
    print("\n✅ Comparison charts saved to eval/results/")

    plt.show()


if __name__ == "__main__":
    compare_agents()