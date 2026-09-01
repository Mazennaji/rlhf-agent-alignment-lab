import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class PreferenceDataset(Dataset):
    def __init__(self, pkl_path: str, action_dim: int, indices=None):
        with open(pkl_path, "rb") as f:
            all_data = pickle.load(f)
        self.data = [all_data[i] for i in indices] if indices is not None else all_data
        self.action_dim = action_dim

    def __len__(self):
        return len(self.data)

    def _to_tensors(self, traj):
        obs = torch.tensor(traj["observations"], dtype=torch.float32)
        actions = traj["actions"].astype(np.int64).reshape(-1)
        actions_onehot = torch.nn.functional.one_hot(
            torch.tensor(actions), num_classes=self.action_dim
        ).float()
        return obs, actions_onehot

    def __getitem__(self, idx):
        pair = self.data[idx]
        obs_a, act_a = self._to_tensors(pair["traj_a"])
        obs_b, act_b = self._to_tensors(pair["traj_b"])
        preferred = torch.tensor(pair["preferred"], dtype=torch.float32)
        return obs_a, act_a, obs_b, act_b, preferred


def collate_fn(batch):
    return batch


def train_val_split(pkl_path: str, val_fraction: float, seed: int):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    n = len(data)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    return train_idx, val_idx