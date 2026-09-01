import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from envs.base_env import make_env
from configs.config import CFG
from dpo.dpo_policy import DPOPolicy


def dpo_predict(policy, obs, deterministic=True):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = policy(obs_t)
        if deterministic:
            action = torch.argmax(logits, dim=-1).item()
        else:
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
    return action


def evaluate_dpo(n_episodes=50, max_steps=500):
    env = make_env(CFG.env_id, CFG.seed + 999)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = DPOPolicy(obs_dim, action_dim)
    policy.load_state_dict(torch.load("dpo/checkpoints/dpo_policy.pt"))
    policy.eval()

    env_returns = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_return = 0.0
        for _ in range(max_steps):
            action = dpo_predict(policy, obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            if terminated or truncated:
                break
        env_returns.append(ep_return)

    env.close()

    print(f"\n--- DPO-Tuned Agent ---")
    print(f"Env reward: mean={np.mean(env_returns):.2f}  std={np.std(env_returns):.2f}")
    return env_returns


if __name__ == "__main__":
    evaluate_dpo()