<div align="center">

# RLHF Agent Alignment Lab

**Learning reward from human preference, not hand-crafted heuristics.**

A from-scratch implementation of the Reinforcement Learning from Human Feedback (RLHF) pipeline — base policy training, preference collection, reward modeling, and KL-regularized fine-tuning — built on Gymnasium and Stable-Baselines3.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Env-0A9EDC?style=for-the-badge&logo=openaigym&logoColor=white)](https://gymnasium.farama.org/)
[![Stable Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-6E56CF?style=for-the-badge)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-000000?style=for-the-badge)](#license)

</div>

---

## Overview

Most reinforcement learning assumes a reward function that fully captures what you want the agent to do. In practice, that reward is often a rough proxy — easy to specify, hard to get right, and easy for a capable agent to exploit.

**RLHF Agent Alignment Lab** demonstrates the alternative: instead of hand-coding a reward function, you *learn* one from human judgments of which behavior is better, then optimize the agent against that learned signal — with guardrails to keep it from drifting into reward-hacking territory.

This is the same three-stage recipe used to align large language models (pretrain → reward model → RL fine-tune with KL penalty), scaled down to a transparent, inspectable RL environment.

<br>

## Pipeline

```
Base Policy (PPO)  →  Preference Collection  →  Reward Model  →  RLHF Fine-Tune  →  Evaluation
```

| Stage | What happens |
|---|---|
| **1. Base agent** | A standard PPO agent is trained on the environment's native reward signal — the untuned starting point. |
| **2. Preference collection** | Pairs of trajectories from the base agent are compared — a human (or simulated oracle) simply indicates which of the two is better. |
| **3. Reward model** | A neural network is trained on those pairwise comparisons using a Bradley–Terry loss, learning to score any trajectory the way a human would. |
| **4. RLHF fine-tuning** | PPO is re-run using the reward model's output in place of the environment reward, with a KL-divergence penalty against the base policy to prevent reward hacking. |
| **5. Evaluation** | The base and RLHF-tuned agents are compared on both raw environment reward and human-judged trajectory quality. |

<br>

## Repository structure

```
rlhf-agent-alignment-lab/
├── base_agent/            PPO agent trained on raw environment reward
├── preference_collection/ Trajectory pairing and preference-labeling scripts
├── reward_model/          Bradley–Terry reward model and training loop
├── rlhf_finetune/         PPO fine-tuning against the learned reward + KL penalty
├── eval/                  Base vs. RLHF-tuned agent comparison
├── notebooks/             Reward model curves, preference accuracy, KL tuning
└── README.md
```

<br>

## Core concepts

- **PPO** — the actor-critic algorithm used for both base training and fine-tuning, relying on a clipped surrogate objective to keep policy updates conservative.
- **Bradley–Terry preference modeling** — converts pairwise "A is better than B" comparisons into a differentiable loss for training the reward model.
- **KL-divergence regularization** — anchors the fine-tuned policy to the base policy, trading off reward-model optimization against behavioral drift.
- **Reward hacking** — the lab deliberately surfaces cases where the agent over-optimizes an imperfect learned reward, since recognizing this failure mode is as instructive as avoiding it.
- **Human-in-the-loop evaluation** — final comparisons are judged on alignment with human preference, not just cumulative environment reward.

<br>

## Getting started

```bash
git clone https://github.com/<your-username>/rlhf-agent-alignment-lab.git
cd rlhf-agent-alignment-lab
pip install -r requirements.txt
```

**Run the pipeline end to end:**

```bash
python base_agent/train.py
python preference_collection/generate_pairs.py
python reward_model/train_reward_model.py
python rlhf_finetune/finetune.py
python eval/compare_agents.py
```

<br>

## Requirements

```
gymnasium
stable-baselines3
torch
numpy
pandas
matplotlib
optuna
```

<br>

## Disclaimer

This project is for educational purposes only. It is a simplified illustration of RLHF mechanics and is not intended for production alignment work, financial use, or any safety-critical application.

<br>

## License

Released under the [MIT License](LICENSE).

<div align="center">

<sub>Built with Gymnasium, Stable-Baselines3, and PyTorch.</sub>

</div>
