import re
from pathlib import Path
import sys
import torch

# ==========================
# RETRIEVAL
# ==========================
def retrieve_character_dialogues(character, file):
    dialogues = []
    with open(file) as f:
        for line in f:
            search_line = line.lower()
            char_found = search_line.find(character)
            comma_found = search_line.find(',,')
            if char_found != -1 and comma_found != -1 and char_found < comma_found:
                dialogues.append(line[comma_found+2:])
    return dialogues

import re

def extract_all_dialogues(file):

    dialogues = set()  # use set to avoid duplicates
    in_events = False
    pattern = re.compile(r",,(\w+)")  # ensures we match lines with ',,<word>'

    with open(file, encoding="utf-8") as f:
        for line in f:
            if not in_events:
                if line.strip().lower() == "[events]":
                    in_events = True
                continue

            if pattern.search(line):  # safer than startswith("Dialogue:")
                parts = line.strip().split(",", 9)
                if len(parts) >= 10:
                    character = parts[4].strip()
                    dialogue = parts[9]

                    # clean dialogue text
                    dialogue = re.sub(r"\{.*?\}", "", dialogue)  # remove {...}
                    dialogue = dialogue.replace("\\N", " ").strip()

                    if character and dialogue:
                        dialogues.add(f"{character} : {dialogue}")

    return list(dialogues)


def load_and_clean_dialogues(character = None , path_pattern="**/*.eng.ass"):
    """
    Loads and cleans dialogues from .ass files.
    - Uses extract_all_dialogues()
    - Deduplicates while preserving order
    - Cleans {tags} and "shock!" but keeps line breaks
    """
    bracket_content = re.compile(r'\{.*?\}|shock!')

    p = Path(".")
    paths = list(p.glob(path_pattern))

    seen = set()
    dialogues = []
    if character is None:
        for path in paths:
            for d in extract_all_dialogues(path):
                if d not in seen:
                    seen.add(d)
                    dialogues.append(d)

        # clean and join dialogues into one text blob
        cleaned = bracket_content.sub("", "\n".join(dialogues))
    else:
        for path in paths:
            for d in retrieve_character_dialogues(character,path):
                if d not in seen:
                    seen.add(d)
                    dialogues.append(d)
        cleaned = bracket_content.sub("", "".join(dialogues))
        
    size_kb = sys.getsizeof(cleaned) // 1024
    print(f"Memory size: {size_kb} KB")
    print(f"Number of dialogues: {len(dialogues)}")
    return cleaned


# ==========================
# ENCODING
# ==========================
def build_vocab(text):
    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Possible Vocabulary = [{''.join(chars)}]")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join(itos[i] for i in l)
    return stoi, itos, encode, decode, vocab_size


def get_splits(data, split_ratio=0.9):
    n = int(split_ratio * len(data))
    return data[:n], data[n:]


# ==========================
# DATA LOADER
# ==========================
def get_batch(split, train_data, val_data, block_size, batch_size, device):
    dataset = train_data if split == "train" else val_data
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i:i+block_size] for i in ix])
    y = torch.stack([dataset[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
