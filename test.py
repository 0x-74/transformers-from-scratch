import torch
from einops import rearrange
from torch.nn import functional as F

from v2 import BigramLanguageModel, decode  # adjust import paths if needed
# ==========================
# LOAD MODEL
# ==========================
DEVICE = 'cuda'
m = BigramLanguageModel().to(DEVICE)
m.load_state_dict(torch.load("gpt.pth", map_location=DEVICE))
m.eval()

# ==========================
# GENERATION
# ==========================
start_token = torch.zeros((1, 1), device=DEVICE, dtype=torch.long)  # single "zero" token
generated = m.generate(start_token, max_new_tokens=1200)[0].tolist()

print(decode(generated))
