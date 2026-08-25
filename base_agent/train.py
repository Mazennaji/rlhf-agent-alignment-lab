import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from envs.base_env import make_env
from configs.config import CFG


def train_base_agent():
    train_env = Monitor(make_env(CFG.env_id, CFG.seed))

    eval_env = Monitor(make_env(CFG.env_id, CFG.seed + 1))

    os.makedirs("base_agent/checkpoints", exist_ok=True)
    os.makedirs("base_agent/logs", exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="base_agent/checkpoints",
        log_path="base_agent/logs",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        seed=CFG.seed,
        tensorboard_log="base_agent/logs/tensorboard",
    )

    model.learn(
        total_timesteps=CFG.base_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save("base_agent/checkpoints/base_ppo_final")
    print("✅ Base agent training complete. Model saved to base_agent/checkpoints/base_ppo_final.zip")

    train_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    train_base_agent()