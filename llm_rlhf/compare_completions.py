import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from llm_rlhf.lm_base import load_base_lm, generate_completion
from llm_rlhf.prompts import PROMPTS


def compare_completions():
    base_model, tokenizer = load_base_lm()
    rlhf_model = GPT2LMHeadModel.from_pretrained("llm_rlhf/checkpoints/rlhf_gpt2")

    for prompt in PROMPTS[:4]:
        base_completion, _ = generate_completion(base_model, tokenizer, prompt, do_sample=False)
        rlhf_completion, _ = generate_completion(rlhf_model, tokenizer, prompt, do_sample=False)

        print(f"\nPrompt: {prompt}")
        print(f"  Base:  {base_completion.strip()}")
        print(f"  RLHF:  {rlhf_completion.strip()}")


if __name__ == "__main__":
    compare_completions()