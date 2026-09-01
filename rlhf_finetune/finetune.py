import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs.base_env import make_env
from configs.config import CFG
from reward_model.ensemble import RewardModelEnsemble
from rlhf_finetune.reward_wrapper import RLHFRewardWrapper, ModelRef
from rlhf_finetune.kl_schedule import linear_kl_schedule


def load_reward_model(obs_dim: int, action_dim: int) -> RewardModelEnsemble:
    return RewardModelEnsemble.load(
        obs_dim, action_dim, n_models=CFG.reward_model_ensemble_size
    )


def finetune_with_rlhf():
    probe_env = make_env(CFG.env_id, CFG.seed)
    obs_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.n
    probe_env.close()

    reward_model = load_reward_model(obs_dim, action_dim)
    base_model = PPO.load("base_agent/checkpoints/base_ppo_final")

    model_ref = ModelRef()
    kl_schedule = linear_kl_schedule(start=0.3, end=0.02, total_steps=CFG.finetune_timesteps)

    raw_env = make_env(CFG.env_id, CFG.seed + 200)
    wrapped_env = RLHFRewardWrapper(
        raw_env,
        reward_model=reward_model,
        base_model=base_model,
        current_model_ref=model_ref,
        action_dim=action_dim,
        kl_coef_fn=kl_schedule,
        total_steps=CFG.finetune_timesteps,
    )
    wrapped_env = Monitor(wrapped_env)

    finetuned_model = PPO.load("base_agent/checkpoints/base_ppo_final", env=wrapped_env)
    model_ref.model = finetuned_model

    finetuned_model.learn(total_timesteps=CFG.finetune_timesteps, progress_bar=True)

    os.makedirs("rlhf_finetune/checkpoints", exist_ok=True)
    finetuned_model.save("rlhf_finetune/checkpoints/rlhf_ppo_final")
    print("✅ RLHF fine-tuning complete. Model saved to rlhf_finetune/checkpoints/rlhf_ppo_final.zip")

    wrapped_env.close()
    return finetuned_model


if __name__ == "__main__":
    finetune_with_rlhf()