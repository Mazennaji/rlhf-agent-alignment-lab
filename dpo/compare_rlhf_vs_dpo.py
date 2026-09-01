import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from envs.base_env import make_env
from configs.config import CFG
from dpo.dpo_policy import DPOPolicy
from dpo.evaluate_dpo import dpo_predict


def evaluate_ppo_agent(model_path, n_episodes=50, max_steps=500):
    env = make_env(CFG.env_id, CFG.seed + 999)
    model = PPO.load(model_path)

    returns = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_return = 0.0
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            if terminated or truncated:
                break
        returns.append(ep_return)
    env.close()
    return returns


def evaluate_dpo_agent(n_episodes=50, max_steps=500):
    env = make_env(CFG.env_id, CFG.seed + 999)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = DPOPolicy(obs_dim, action_dim)
    policy.load_state_dict(torch.load("dpo/checkpoints/dpo_policy.pt"))
    policy.eval()

    returns = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_return = 0.0
        for _ in range(max_steps):
            action = dpo_predict(policy, obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            if terminated or truncated:
                break
        returns.append(ep_return)
    env.close()
    return returns


def compare_rlhf_vs_dpo():
    base_returns = evaluate_ppo_agent("base_agent/checkpoints/base_ppo_final")
    rlhf_returns = evaluate_ppo_agent("rlhf_finetune/checkpoints/rlhf_ppo_final")
    dpo_returns = evaluate_dpo_agent()

    print(f"Base PPO:   mean={np.mean(base_returns):.2f}  std={np.std(base_returns):.2f}")
    print(f"RLHF (PPO): mean={np.mean(rlhf_returns):.2f}  std={np.std(rlhf_returns):.2f}")
    print(f"DPO:        mean={np.mean(dpo_returns):.2f}  std={np.std(dpo_returns):.2f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(
        [base_returns, rlhf_returns, dpo_returns],
        tick_labels=["Base PPO", "RLHF (PPO)", "DPO"],
    )
    ax.set_title("Base vs. RLHF vs. DPO — Env Reward")
    ax.set_ylabel("Episode Return")

    os.makedirs("dpo/results", exist_ok=True)
    fig.savefig("dpo/results/rlhf_vs_dpo.png")
    print("✅ Chart saved to dpo/results/rlhf_vs_dpo.png")
    plt.show()


if __name__ == "__main__":
    compare_rlhf_vs_dpo()