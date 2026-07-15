import torch

# tells python to save text as strings first instead of 
# evaluating them immediately
from __future__ import annotations

import math
import os
import argparse

from main import val_data
 
def build_tokenizer(data_path: str = "input.txt"):
    with open(data_path, "r", encoding="utf-8") as f: # opens the file in read mode with utf-8 encoding
        text = f.read() # reads the entire file into a string
    
    # saving text in a list of strings
    chars = sorted(list(set(text)))
    char_to_index = {
        ch: i for i, ch in enumerate(chars)
    }
    index_to_char = {
        i: ch for i, ch in enumerate(chars)
    }

    # encode and decode the passed text
    def encode(string: str):
        return [char_to_index[c] for c in string if c in char_to_index]

    def decode(tokens):
        return "".join(index_to_char[i] for i in tokens if i in index_to_char)
    
    data = torch.tensor(encode(text), dtype=torch.long) # keep it floating
    split = int(0.9 * len(data))
    # first 90% for training, last 10% for validation
    train_data = data[:split]
    val_data = data[split:]

    # return the tokenizer object
    return {
        "encode": encode,
        "decode": decode,
        "train_data": train_data,
        "val_data": val_data,
        "vocab_size": len(chars),
    }