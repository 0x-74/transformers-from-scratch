# Differences : 
# creates an N dimensional embedding and then uses a linear layer to map the embeddings 
# to vocab_size to get the logits 
import re
from pathlib import Path
import sys
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F


# ==========================
# HYPERPARAMETERS
# ==========================
CHARACTER = 'anya'
BLOCK_SIZE = 8
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
LEARNING_RATE = 1e-2
TRAIN_STEPS = 10_000
EVAL_INTERVAL = 1_000
EVAL_ITERS = 200
SEED = 74
N_EMBD = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ==========================


def retrieve_character_dialogues(character, file):
    dialogues = []
    with open(file) as f:
        for line in f:
            search_line = line.lower()
            anya_found = search_line.find(character)
            comma_found = search_line.find(',,')
            if anya_found != -1 and comma_found != -1 and anya_found < comma_found:
                dialogues.append(line[comma_found+2:])
    return dialogues


# Regex for cleaning
bracket_content = re.compile(r'\{.*?\}|shock!|\n')

# Collect dialogues
p = Path('.')
paths = list(p.glob('**/*.eng.ass'))
dialogues = []
for path in paths:
    dialogues.extend(retrieve_character_dialogues(CHARACTER, path))
cleaned_dialogues = bracket_content.sub("", "".join(dialogues))

# Memory and stats
size_kb = sys.getsizeof(cleaned_dialogues) // 1024
print(f"Memory size: {size_kb} KB")
print(f"Number of dialogues: {len(cleaned_dialogues)}")

chars = sorted(set(cleaned_dialogues))
VOCAB_SIZE = len(chars)
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Possible Vocabulary = [{''.join(chars)}]")

# Encoding/decoding
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

# Test encode/decode
encoded_text = encode("Hehe")
print("encoded text:", encoded_text)
decoded_text = decode(encoded_text)
print("decoded text:", decoded_text)

# Convert to tensor
data = torch.tensor(encode(cleaned_dialogues), dtype=torch.long)
print(data.shape, data.dtype)

# Train/val split
n = int(TRAIN_SPLIT * len(data))
train_data = data[:n]
val_data = data[n:]


# ==========================
# DATA LOADER
# ==========================
torch.manual_seed(SEED)

def get_batch(split):
    dataset = train_data if split == "train" else val_data
    ix = torch.randint(len(dataset) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([dataset[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([dataset[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ==========================
# MODEL
# ==========================
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.lm_head = nn.Linear(N_EMBD,VOCAB_SIZE)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE,N_EMBD)
    def forward(self, idx, targets=None):
        B,T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T,device=DEVICE))
        x = tok_emb+pos_emb
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            logits = rearrange(logits, "b t c -> (b t) c")
            targets = rearrange(targets, "b t -> (b t)")
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==========================
# EVALUATION FUNCTION
# ==========================
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
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
    # Evaluate periodically
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch and train
    xb, yb = get_batch("train")
    _, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Final loss:", loss.item())

# ==========================
# SAMPLE GENERATION
# ==========================
start_token = torch.zeros((1, 1), device=DEVICE, dtype=torch.long)
print(decode(m.generate(start_token, max_new_tokens=50)[0].tolist()))
