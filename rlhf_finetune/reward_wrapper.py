import gymnasium as gym
import torch


class ModelRef:
    def __init__(self):
        self.model = None


class RLHFRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_model, base_model, current_model_ref: ModelRef,
                 action_dim: int, kl_coef: float = 0.1, kl_coef_fn=None, total_steps=None):
        super().__init__(env)
        self.reward_model = reward_model
        self.base_model = base_model
        self.current_model_ref = current_model_ref
        self.action_dim = action_dim
        self.kl_coef = kl_coef
        self.kl_coef_fn = kl_coef_fn
        self.total_steps = total_steps
        self.step_count = 0
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def _current_kl_coef(self):
        if self.kl_coef_fn is not None:
            return self.kl_coef_fn(self.step_count)
        return self.kl_coef

    def step(self, action):
        next_obs, env_reward, terminated, truncated, info = self.env.step(action)

        obs_t = torch.tensor(self._last_obs, dtype=torch.float32).unsqueeze(0)
        action_onehot = torch.nn.functional.one_hot(
            torch.tensor([action]), num_classes=self.action_dim
        ).float()

        with torch.no_grad():
            learned_reward = self.reward_model(obs_t, action_onehot).item()

        kl_penalty = 0.0
        current_model = self.current_model_ref.model
        if current_model is not None:
            with torch.no_grad():
                obs_tensor = current_model.policy.obs_to_tensor(self._last_obs[None])[0]
                base_dist = self.base_model.policy.get_distribution(obs_tensor)
                current_dist = current_model.policy.get_distribution(obs_tensor)
                kl_penalty = torch.distributions.kl_divergence(
                    current_dist.distribution, base_dist.distribution
                ).mean().item()

        kl_coef = self._current_kl_coef()
        shaped_reward = learned_reward - kl_coef * kl_penalty

        info["env_reward"] = env_reward
        info["learned_reward"] = learned_reward
        info["kl_penalty"] = kl_penalty
        info["kl_coef"] = kl_coef

        self.step_count += 1
        self._last_obs = next_obs
        return next_obs, shaped_reward, terminated, truncated, info