import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import pickle
import torch
import numpy as np
from stable_baselines3 import PPO

from envs.base_env import make_env
from configs.config import CFG
from dpo.dpo_policy import DPOPolicy


def build_policy_from_ppo(ppo_model, obs_dim, action_dim):
    policy = DPOPolicy(obs_dim, action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    env = make_env(CFG.env_id, CFG.seed + 900)
    obs_buffer, action_buffer = [], []

    for _ in range(50):
        obs, info = env.reset()
        for _ in range(CFG.trajectory_length):
            action, _ = ppo_model.predict(obs, deterministic=False)
            obs_buffer.append(obs)
            action_buffer.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    env.close()

    obs_t = torch.tensor(np.array(obs_buffer), dtype=torch.float32)
    action_t = torch.tensor(np.array(action_buffer), dtype=torch.long)

    for _ in range(200):
        logits = policy(obs_t)
        loss = torch.nn.functional.cross_entropy(logits, action_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return policy


def dpo_loss(policy, ref_policy, obs_a, act_a, obs_b, act_b, preferred, beta=0.1):
    policy_logp_a = policy.trajectory_log_prob(obs_a, act_a)
    policy_logp_b = policy.trajectory_log_prob(obs_b, act_b)

    with torch.no_grad():
        ref_logp_a = ref_policy.trajectory_log_prob(obs_a, act_a)
        ref_logp_b = ref_policy.trajectory_log_prob(obs_b, act_b)

    policy_ratio_diff = (policy_logp_a - ref_logp_a) - (policy_logp_b - ref_logp_b)

    if preferred.item() == 0:
        policy_ratio_diff = -policy_ratio_diff

    loss = -torch.nn.functional.logsigmoid(beta * policy_ratio_diff)
    return loss


def prepare_dpo_batch(pair, action_dim):
    traj_a, traj_b = pair["traj_a"], pair["traj_b"]
    obs_a = torch.tensor(traj_a["observations"], dtype=torch.float32)
    act_a = torch.tensor(traj_a["actions"].astype(np.int64).reshape(-1), dtype=torch.long)
    obs_b = torch.tensor(traj_b["observations"], dtype=torch.float32)
    act_b = torch.tensor(traj_b["actions"].astype(np.int64).reshape(-1), dtype=torch.long)
    preferred = torch.tensor(pair["preferred"], dtype=torch.float32)
    return obs_a, act_a, obs_b, act_b, preferred


def train_dpo(beta=0.1, epochs=10, lr=1e-4):
    probe_env = make_env(CFG.env_id, CFG.seed)
    obs_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.n
    probe_env.close()

    ppo_base = PPO.load("base_agent/checkpoints/base_ppo_final")
    policy = build_policy_from_ppo(ppo_base, obs_dim, action_dim)
    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    with open("preference_collection/data/preference_pairs.pkl", "rb") as f:
        pairs = pickle.load(f)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        np.random.shuffle(pairs)

        for pair in pairs:
            obs_a, act_a, obs_b, act_b, preferred = prepare_dpo_batch(pair, action_dim)
            loss = dpo_loss(policy, ref_policy, obs_a, act_a, obs_b, act_b, preferred, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[DPO] epoch {epoch+1}/{epochs} loss={total_loss / len(pairs):.4f}")

    os.makedirs("dpo/checkpoints", exist_ok=True)
    torch.save(policy.state_dict(), "dpo/checkpoints/dpo_policy.pt")
    print("✅ DPO training complete. Saved to dpo/checkpoints/dpo_policy.pt")

    return policy


if __name__ == "__main__":
    train_dpo(beta=0.1, epochs=10)