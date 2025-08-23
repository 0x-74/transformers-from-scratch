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


# ---------------------------------------------------------------------------
# Dataset Creation
# ---------------------------------------------------------------------------

def retrieve_character_dialogues(character: str, file: Path):
    """Extract dialogues for a given character from .ass subtitle file."""
    dialogues = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            search_line = line.lower()
            anya_found = search_line.find(character)
            comma_found = search_line.find(",,")
            if anya_found != -1 and comma_found != -1 and anya_found < comma_found:
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
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
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
# Batch Loader
# ---------------------------------------------------------------------------

torch.manual_seed(74)
batch_size = 4
block_size = 8

def get_batch(split: str):
    """Return a batch of data (x, y) for training/validation."""
    dataset = train_data if split == "train" else val_data
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i : i + block_size] for i in ix])
    y = torch.stack([dataset[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# Visualize batch
xb, yb = get_batch("train")
print("Inputs (xb):", xb.shape)
print("Targets (yb):", yb.shape)


# ---------------------------------------------------------------------------
# Bigram Language Model
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
# Training
# ---------------------------------------------------------------------------

m = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-2)

batch_size = 32
for step in range(10_000):
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Final loss:", loss.item())


# ---------------------------------------------------------------------------
# Text Generation
# ---------------------------------------------------------------------------

output = m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)
print(decode(output[0].tolist()))
