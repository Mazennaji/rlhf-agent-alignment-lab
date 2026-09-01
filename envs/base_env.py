import gymnasium as gym

def make_env(env_id: str = "LunarLander-v3", seed: int = 0, render_mode: str = None):
    env = gym.make(env_id, render_mode=render_mode) if render_mode else gym.make(env_id)
    env.reset(seed=seed)
    return env