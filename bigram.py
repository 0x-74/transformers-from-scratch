#!/usr/bin/env python
# coding: utf-8

"""
Bigram Language Model training on character dialogues extracted from .ass subtitle files.
"""

import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from model_utils import train_loop
from profiler_utils import log_generation  # our reusable training function

# ---------------------------------------------------------------------------
# Dataset Creation
# ---------------------------------------------------------------------------

def retrieve_character_dialogues(character: str, file: Path):
    """Extract dialogues for a given character from .ass subtitle file."""
    dialogues = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            search_line = line.lower()
            char_found = search_line.find(character)
            comma_found = search_line.find(",,")
            if char_found != -1 and comma_found != -1 and char_found < comma_found:
                dialogues.append(line[comma_found + 2 :])
    return dialogues


# Compile regex for removing formatting/bracket content
bracket_content = re.compile(r"\{.*?\}|shock!|\n")

# Collect all .ass files
p = Path(".")
paths = list(p.glob("**/*.eng.ass"))
character = "anya"

# Extract and clean dialogues
dialogues = []
for path in paths:
    dialogues.extend(retrieve_character_dialogues(character, path))

cleaned_dialogues = bracket_content.sub("", "".join(dialogues))


# ---------------------------------------------------------------------------
# Preliminary Checks
# ---------------------------------------------------------------------------

size_kb = sys.getsizeof(cleaned_dialogues) // 1024
print(f"Memory size: {size_kb} KB")
print(f"Number of dialogues: {len(cleaned_dialogues)}")

chars = sorted(set(cleaned_dialogues))
VOCAB_SIZE = len(chars)
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Possible Vocabulary = [{''.join(chars)}]")


# ---------------------------------------------------------------------------
# Encoder and Decoder Functions
# ---------------------------------------------------------------------------

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

# Quick test
encoded_text = encode("Hehe")
print("encoded text:", encoded_text)
print("decoded text:", decode(encoded_text))


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

data = torch.tensor(encode(cleaned_dialogues), dtype=torch.long)
print(data.shape, data.dtype)

# Train/Val split
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------

class BigramLanguageModel(nn.Module):
    """A simple bigram language model."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            logits = rearrange(logits, "b t c -> (b t) c")
            targets = rearrange(targets, "b t -> (b t)")
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        """Generate new text from given indices."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# Training Orchestration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Hyperparameters
    from config import *

    torch.manual_seed(SEED)

    model = BigramLanguageModel(VOCAB_SIZE).to(DEVICE)

    # Train model with reusable train_loop
    model, final_loss = train_loop(
        model=model,
        device=DEVICE,
        lr=LEARNING_RATE,
        steps=TRAIN_STEPS,
        eval_interval=EVAL_INTERVAL,
        eval_iters=EVAL_ITERS,
        train_data=train_data,
        val_data=val_data,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    # -----------------------------------------------------------------------
    # Text Generation
    # -----------------------------------------------------------------------
    start_token = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    sample = log_generation(model, decode, start_token, max_new_tokens=500)
    print(sample)

    # Save model
    torch.save(model.state_dict(), "bigram.pth")
