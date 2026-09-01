import os
from dataclasses import dataclass, fields


@dataclass
class Config:
    env_id: str = "LunarLander-v3"
    seed: int = 0

    base_timesteps: int = 200_000

    num_trajectory_pairs: int = 500
    trajectory_length: int = 300
    min_pair_diversity: float = 0.15

    reward_model_epochs: int = 20
    reward_model_lr: float = 3e-4
    reward_model_ensemble_size: int = 5
    reward_model_val_split: float = 0.15

    finetune_timesteps: int = 80_000
    kl_coef: float = 0.1

    def __post_init__(self):
        for f in fields(self):
            env_val = os.getenv(f.name.upper())
            if env_val is not None:
                setattr(self, f.name, f.type(env_val))


CFG = Config()