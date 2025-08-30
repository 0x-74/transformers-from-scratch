import torch
from torch import nn, optim
import sys, os

from data_utils import get_batch
from profiler_utils import record_time_function

LOG_FILE = "train_losses.log"

def log_message(msg: str):
    """Helper: print + append to log file."""
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    train_data,
    val_data,
    block_size: int,
    batch_size: int,
    device: torch.device,
    eval_iters: int,
):
    """Estimate average train/val loss."""
    model.eval()
    out = {}

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    return out


@record_time_function(runs=1, warmup=0)
def train_loop(
    model: nn.Module,
    device: torch.device,
    lr: float,
    steps: int,
    eval_interval: int,
    eval_iters: int,
    train_data=None,
    val_data=None,
    block_size: int = 128,
    batch_size: int = 32,
    seed: int = 74,
):
    """
    Train a language model.
    Logs losses along with the entrypoint filename.
    """
    torch.manual_seed(seed)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    final_loss = None

    # Get the entrypoint script (the file originally executed)
    caller_file = os.path.basename(sys.argv[0]) if sys.argv else "<unknown>"

    for step in range(steps):
        if step % eval_interval == 0:
            losses = evaluate_loss(
                model, train_data, val_data, block_size, batch_size, device, eval_iters
            )
            msg = (
                f"[{caller_file}] Step {step} | "
                f"train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f} | "
            )
            log_message(msg)

        xb, yb = get_batch("train", train_data, val_data, block_size, batch_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()

    msg = f"[{caller_file}] Final loss: {final_loss:.4f}"
    log_message(msg)

    return model, final_loss
