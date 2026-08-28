import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from stable_baselines3 import PPO

from envs.base_env import make_env
from configs.config import CFG
from reward_model.model import RewardModel
from rlhf_finetune.reward_wrapper import RLHFRewardWrapper, ModelRef


def inspect():
    probe_env = make_env(CFG.env_id, CFG.seed)
    obs_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.n
    probe_env.close()

    reward_model = RewardModel(obs_dim, action_dim)
    reward_model.load_state_dict(torch.load("reward_model/checkpoints/reward_model.pt"))
    reward_model.eval()

    base_model = PPO.load("base_agent/checkpoints/base_ppo_final")
    model_ref = ModelRef()
    model_ref.model = base_model

    raw_env = make_env(CFG.env_id, CFG.seed + 200)
    wrapped_env = RLHFRewardWrapper(
        raw_env, reward_model, base_model, model_ref, action_dim, CFG.kl_coef
    )

    obs, info = wrapped_env.reset()
    for _ in range(5):
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        print(info)

    wrapped_env.close()


if __name__ == "__main__":
    inspect()