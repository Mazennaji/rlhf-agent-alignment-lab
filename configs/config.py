from dataclasses import dataclass

@dataclass
class Config:
    env_id: str = "LunarLander-v3"
    seed: int = 0

    base_timesteps: int = 100_000

    num_trajectory_pairs: int = 500
    trajectory_length: int = 200

    reward_model_epochs: int = 20
    reward_model_lr: float = 3e-4

    finetune_timesteps: int = 50_000
    kl_coef: float = 0.1

CFG = Config()