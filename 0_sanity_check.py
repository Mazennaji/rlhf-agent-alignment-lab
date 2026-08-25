from envs.base_env import make_env
from configs.config import CFG

env = make_env(CFG.env_id, CFG.seed)
obs, info = env.reset()
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"reward={reward:.2f} terminated={terminated} truncated={truncated}")
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("✅ Environment is wired up correctly.")