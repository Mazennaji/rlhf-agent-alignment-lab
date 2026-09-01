import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer


class LMRewardModel(nn.Module):

    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        outputs = self.backbone(**inputs)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        score = self.head(last_hidden)
        return score.squeeze()