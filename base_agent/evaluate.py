import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from envs.base_env import make_env
from configs.config import CFG

env = make_env(CFG.env_id, CFG.seed)
model = PPO.load("base_agent/checkpoints/base_ppo_final")

obs, info = env.reset()
total_reward = 0
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Episode reward: {total_reward:.2f}")
env.close()