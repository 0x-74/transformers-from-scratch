import os
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from data_utils import load_and_clean_dialogues, build_vocab, get_splits, get_batch

# ==========================
# HYPERPARAMETERS
# ==========================
from config import *
from model_utils import train_loop
from profiler_utils import log_generation
# ==========================


# ==========================
# DATA PREP
# ==========================
cleaned_dialogues = load_and_clean_dialogues(CHARACTER)
stoi, itos, encode, decode, VOCAB_SIZE = build_vocab(cleaned_dialogues)

data = torch.tensor(encode(cleaned_dialogues), dtype=torch.long)
train_data, val_data = get_splits(data, TRAIN_SPLIT)


# ==========================
# MODEL
# ==========================
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        # ensure positions live on same device as idx
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )
        x = tok_emb + pos_emb
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            logits = rearrange(logits, "b t c -> (b t) c")
            targets = rearrange(targets, "b t -> (b t)")
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]  # crop context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # last timestep
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==========================
# TRAINING
# ==========================
# ---------------------------------------------------------------------------
# Training Orchestration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Hyperparameters
    from config import *

    torch.manual_seed(SEED)

    model = BigramLanguageModel().to(DEVICE)

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
    torch.save(model.state_dict(), f"{os.path.splitext(os.path.basename(__file__))[0]}.pth")