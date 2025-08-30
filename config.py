import torch

# ==========================
# HYPERPARAMETERS
# ==========================
CHARACTER = "anya"
BLOCK_SIZE = 256
BATCH_SIZE = 64
TRAIN_SPLIT = 0.8
LEARNING_RATE = 3e-4
TRAIN_STEPS = 5_000
EVAL_INTERVAL = 200
EVAL_ITERS = 200
DROPOUT = 0.2
SEED = 74
N_EMBD = 384
HEADS = 4
BLOCKS =  3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
