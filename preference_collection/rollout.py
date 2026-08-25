import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO
from envs.base_env import make_env
from configs.config import CFG


def collect_trajectory(model, env, max_steps, deterministic=False):
    obs, info = env.reset()
    observations, actions, rewards = [], [], []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, info = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

        obs = next_obs
        if terminated or truncated:
            break

    return {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "total_return": float(np.sum(rewards)),
    }


def generate_pair(model, env, max_steps):
    traj_a = collect_trajectory(model, env, max_steps, deterministic=False)
    traj_b = collect_trajectory(model, env, max_steps, deterministic=False)
    return traj_a, traj_b