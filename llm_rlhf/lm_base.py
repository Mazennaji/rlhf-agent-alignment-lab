import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_base_lm(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer


def generate_completion(model, tokenizer, prompt, max_new_tokens=40, do_sample=True, temperature=1.0):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    completion = full_text[len(prompt):]
    return completion, output_ids[0]