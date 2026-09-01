import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import torch
import torch.nn as nn

from llm_rlhf.reward_model import LMRewardModel


def bradley_terry_loss(score_a, score_b, preferred):
    logits = (score_a - score_b).unsqueeze(0)
    return nn.functional.binary_cross_entropy_with_logits(logits, preferred.unsqueeze(0))


def train_lm_reward_model(epochs=3, lr=1e-5):
    with open("llm_rlhf/data/lm_preference_pairs.pkl", "rb") as f:
        pairs = pickle.load(f)

    model = LMRewardModel()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)
    for p in model.backbone.parameters():
        p.requires_grad = False

    for epoch in range(epochs):
        total_loss, correct = 0.0, 0
        for pair in pairs:
            text_a = pair["prompt"] + pair["completion_a"]
            text_b = pair["prompt"] + pair["completion_b"]
            preferred = torch.tensor(float(pair["preferred"]))

            score_a = model(text_a)
            score_b = model(text_b)
            loss = bradley_terry_loss(score_a, score_b, preferred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = 1 if score_a.item() >= score_b.item() else 0
            correct += int(predicted == int(preferred.item()))

        print(f"[LM reward model] epoch {epoch+1}/{epochs} loss={total_loss/len(pairs):.4f} acc={correct/len(pairs):.2%}")

    os.makedirs("llm_rlhf/checkpoints", exist_ok=True)
    torch.save(model.head.state_dict(), "llm_rlhf/checkpoints/reward_head.pt")
    print("✅ Reward model head saved to llm_rlhf/checkpoints/reward_head.pt")
    return model


if __name__ == "__main__":
    train_lm_reward_model()