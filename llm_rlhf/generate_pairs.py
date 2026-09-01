import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import random
from llm_rlhf.lm_base import load_base_lm, generate_completion
from llm_rlhf.prompts import PROMPTS


def simulated_lm_oracle(completion_a, completion_b):
    def quality_score(text):
        words = text.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        return len(words) * unique_ratio

    return 1 if quality_score(completion_a) >= quality_score(completion_b) else 0


def build_lm_preference_dataset(n_pairs=200):
    model, tokenizer = load_base_lm()
    dataset = []

    for i in range(n_pairs):
        prompt = random.choice(PROMPTS)
        completion_a, ids_a = generate_completion(model, tokenizer, prompt, do_sample=True, temperature=1.0)
        completion_b, ids_b = generate_completion(model, tokenizer, prompt, do_sample=True, temperature=1.0)

        preferred = simulated_lm_oracle(completion_a, completion_b)

        dataset.append({
            "prompt": prompt,
            "completion_a": completion_a,
            "completion_b": completion_b,
            "preferred": preferred,
        })

        if (i + 1) % 25 == 0:
            print(f"Generated {i + 1}/{n_pairs} LM preference pairs")

    os.makedirs("llm_rlhf/data", exist_ok=True)
    with open("llm_rlhf/data/lm_preference_pairs.pkl", "wb") as f:
        pickle.dump(dataset, f)

    print(f"✅ Saved {len(dataset)} pairs to llm_rlhf/data/lm_preference_pairs.pkl")
    return dataset


if __name__ == "__main__":
    build_lm_preference_dataset(n_pairs=200)