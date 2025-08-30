import torch
import inspect
import os

LOG_FILE = "timings.log"

def record_time_function(runs=100, warmup=50):
    """
    Decorator to measure average execution time of a function on CUDA.
    Logs both to stdout and a log file.
    Uses the caller's file (where the decorated function is used).
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # --- Get caller file (where the function is used) ---
            stack = inspect.stack()
            caller_frame = stack[1]  # 0 = wrapper, 1 = immediate caller
            caller_file = os.path.basename(caller_frame.filename)

            def log(msg):
                print(msg)
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

            if not torch.cuda.is_available():
                log(f"[WARN] CUDA not available. Timing for {func.__name__} "
                    f"(called from {caller_file}) skipped.")
                return func(*args, **kwargs)

            # Warmup
            for _ in range(warmup):
                func(*args, **kwargs)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            total_time = 0.0
            output = None

            for _ in range(runs):
                start.record()
                output = func(*args, **kwargs)
                end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)  # milliseconds

            avg_time = total_time / runs
            log(f"[CUDA] {func.__name__} (called from {caller_file}): "
                f"{avg_time:.4f} ms (avg over {runs} runs)")
            return output
        return wrapper
    return decorator

import sys

def log_generation(model, decode, start_token, max_new_tokens=500, log_file="generations.log"):
    # The filename being executed (not the caller)
    current_file = sys.argv[0]

    # Generate text
    generated = decode(model.generate(start_token, max_new_tokens)[0].tolist())

    # Print to stdout
    print(f"[{current_file}] {generated}")

    # Append to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- Generation from {current_file} ---\n")
        f.write(generated + "\n")
    return generated