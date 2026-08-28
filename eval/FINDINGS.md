# Findings — RLHF Agent Alignment Lab

## Setup
- Environment: `LunarLander-v3`
- Fine-tune kl_coef: `0.1`
- Fine-tune timesteps: `50000`
- Evaluation episodes per agent: `50`

## Results

| Agent | Env Reward (mean ± std) | Reward Model Score (mean ± std) | Episode Length (mean) |
|---|---|---|---|
| Base PPO | 119.85 ± 103.17 | 79.43 ± 26.10 | 454.0 |
| RLHF-Tuned | 117.12 ± 51.93 | 163.89 ± 75.57 | 469.9 |

## Deltas (RLHF − Base)
- Env reward delta: `-2.74`
- Reward model score delta: `+84.46`

## Interpretation
Possible reward hacking — reward-model score rose while true env reward dropped.

## Charts
- `env_reward_comparison.png`
- `reward_model_comparison.png`
- `episode_length_comparison.png`
- `env_vs_learned_scatter.png`
- `distribution_overlay.png`

## Next steps / limitations
- Preference dataset size and quality directly bound reward model reliability.
- Simulated oracle labels (true return) make this task partly circular; manual labels add genuine signal.
- Short fine-tune duration limits how far the policy can shift from the base agent.
