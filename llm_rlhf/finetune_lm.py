import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import torch
import torch.nn.functional as F

from llm_rlhf.lm_base import load_base_lm
from llm_rlhf.reward_model import LMRewardModel
from llm_rlhf.prompts import PROMPTS


def token_log_probs_from_logits(logits, tokens):
    logits = logits[:, :-1, :]
    targets = tokens[:, 1:]
    lp = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view(targets.shape)
    return lp


def finetune_lm_rlhf(n_steps=200, kl_coef=0.3, lr=1e-5, max_new_tokens=15, max_grad_norm=1.0):
    model, tokenizer = load_base_lm()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    ref_model, _ = load_base_lm()
    ref_model.config.use_cache = False
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    reward_model = LMRewardModel()
    reward_model.head.load_state_dict(torch.load("llm_rlhf/checkpoints/reward_head.pt"))
    reward_model.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    baseline = 0.0
    baseline_momentum = 0.9

    for step in range(n_steps):
        prompt = random.choice(PROMPTS)
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        model.config.use_cache = True
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        model.config.use_cache = False

        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        with torch.no_grad():
            reward = reward_model(full_text).item()

        current_logits = model(output_ids).logits
        cur_lp = token_log_probs_from_logits(current_logits, output_ids)

        with torch.no_grad():
            ref_logits = ref_model(output_ids).logits
            ref_lp = token_log_probs_from_logits(ref_logits, output_ids)

        log_ratio = ref_lp - cur_lp
        kl = (log_ratio.exp() - 1 - log_ratio).mean()

        gen_lp = cur_lp[:, prompt_len - 1:]
        sequence_log_prob = gen_lp.sum()

        baseline = baseline_momentum * baseline + (1 - baseline_momentum) * reward
        advantage = reward - baseline

        loss = -sequence_log_prob * advantage + kl_coef * kl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"step {step+1}/{n_steps} reward={reward:.3f} kl={kl.item():.4f} loss={loss.item():.4f}")

    os.makedirs("llm_rlhf/checkpoints", exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained("llm_rlhf/checkpoints/rlhf_gpt2", safe_serialization=False)
    tokenizer.save_pretrained("llm_rlhf/checkpoints/rlhf_gpt2")
    print("RLHF-tuned GPT-2 saved to llm_rlhf/checkpoints/rlhf_gpt2")

    return model


if __name__ == "__main__":
    finetune_lm_rlhf(n_steps=200)