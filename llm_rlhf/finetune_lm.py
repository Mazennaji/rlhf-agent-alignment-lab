import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import torch
import torch.nn.functional as F

from llm_rlhf.lm_base import load_base_lm
from llm_rlhf.reward_model import LMRewardModel
from llm_rlhf.prompts import PROMPTS
import random


def compute_kl_penalty(logits_current, logits_ref):
    log_probs_current = F.log_softmax(logits_current, dim=-1)
    log_probs_ref = F.log_softmax(logits_ref, dim=-1)
    probs_current = log_probs_current.exp()
    kl = (probs_current * (log_probs_current - log_probs_ref)).sum(-1)
    return kl.mean()


def finetune_lm_rlhf(n_steps=200, kl_coef=0.05, lr=1e-5):
    model, tokenizer = load_base_lm()
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    reward_model = LMRewardModel()
    reward_model.head.load_state_dict(torch.load("llm_rlhf/checkpoints/reward_head.pt"))
    reward_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(n_steps):
        prompt = random.choice(PROMPTS)
        inputs = tokenizer(prompt, return_tensors="pt")

        output_ids = model.generate(
            **inputs, max_new_tokens=30, do_sample=True, temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        with torch.no_grad():
            reward = reward_model(full_text).item()

        current_logits = model(output_ids).logits
        with torch.no_grad():
            ref_logits = ref_model(output_ids).logits

        kl = compute_kl_penalty(current_logits, ref_logits)

        target = torch.tensor(reward - kl_coef * kl.item())
        log_probs = F.log_softmax(current_logits[:, :-1, :], dim=-1)
        token_log_probs = log_probs.gather(2, output_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        sequence_log_prob = token_log_probs.sum()

        loss = -sequence_log_prob * target.detach()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"step {step+1}/{n_steps} reward={reward:.3f} kl={kl.item():.4f} loss={loss.item():.4f}")

    os.makedirs("llm_rlhf/checkpoints", exist_ok=True)
    model.save_pretrained("llm_rlhf/checkpoints/rlhf_gpt2")
    tokenizer.save_pretrained("llm_rlhf/checkpoints/rlhf_gpt2")
    print("✅ RLHF-tuned GPT-2 saved to llm_rlhf/checkpoints/rlhf_gpt2")

    return model


if __name__ == "__main__":
    finetune_lm_rlhf(n_steps=200)