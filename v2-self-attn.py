import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from data_utils import load_and_clean_dialogues, build_vocab, get_splits, get_batch

# ==========================
# HYPERPARAMETERS
# ==========================
torch.manual_seed(74)
CHARACTER = 'anya'
BLOCK_SIZE = 8
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
LEARNING_RATE = 1e-3
TRAIN_STEPS = 5_000
EVAL_INTERVAL = 1_000
EVAL_ITERS = 200
SEED = 74
N_EMBD = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.sa_head = Head(N_EMBD)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.sa_head(x)
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
m = BigramLanguageModel().to(DEVICE)
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

for step in range(TRAIN_STEPS):
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss(m)
        print(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train", train_data, val_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
    _, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Final loss:", loss.item())


# ==========================
# SAMPLE GENERATION
# ==========================
start_token = torch.zeros((1, 1), device=DEVICE, dtype=torch.long)
print(decode(m.generate(start_token, max_new_tokens=500)[0].tolist()))
