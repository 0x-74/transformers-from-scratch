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
# DATA PREP
# ==========================
cleaned_dialogues = load_and_clean_dialogues()
stoi, itos, encode, decode, VOCAB_SIZE = build_vocab(cleaned_dialogues)

data = torch.tensor(encode(cleaned_dialogues), dtype=torch.long)
train_data, val_data = get_splits(data, TRAIN_SPLIT)


# ==========================
# MODEL
# ==========================
class FeedForward(nn.Module):
    """linear layer followed by a non linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd),
            nn.ReLU(),
            nn.Linear(4* n_embd,n_embd),
    
        )
    def forward(self,x):
        return self.net(x)
    
class Head(nn.Module):
    """a single head of self-attention"""
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2,-1) * C **-0.5 # (B,T,C) x (B,C,T) = (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0 , float('-inf'))
        wei = F.softmax(wei,dim=-1)
        
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD,N_EMBD)
        
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class Block(nn.Module):
    def __init__(self, n_embd,n_heads):
        super().__init__()
        head_size = n_embd//n_heads
        self.ma = MultiHeadAttention(n_heads,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self,x):
        x = x + self.ma(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(Block(N_EMBD,n_heads=4)
                                   ,Block(N_EMBD,n_heads=4)
                                   ,Block(N_EMBD,n_heads=4)
                                   ,nn.LayerNorm(N_EMBD))
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            logits = rearrange(logits, "b t c -> (b t) c")
            targets = rearrange(targets, "b t -> (b t)")
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:,-BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================
# EVALUATION FUNCTION
# ==========================
@torch.no_grad()
def estimate_loss(m):
    out = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            _, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    m.train()
    return out


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