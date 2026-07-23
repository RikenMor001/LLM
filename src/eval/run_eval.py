import torch

# tells python to save text as strings first instead of 
# evaluating them immediately
from __future__ import annotations

import math
import os
import argparse

from config import CHECKPOINT_PATH, CONTEXT_LENGTH, DATA_PATH
from main import sample_next_token
from sampling import SamplingParams
import sampling

def build_tokenizer(data_path: str = DATA_PATH):
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

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# EVALS

EVAL_PROMPTS = [
    "Once upon a time",
    "Revenue increased",
    "The afternoon sun",
    "Risk factors include"
]

@torch.no_grad()
def estimate_loss(model, get_batch, n_batches: int = 40):
    model.eval()
    losses = {}

    for split in ("train", "val"):
        split_losses = []
        for _ in range(n_batches):
            x, y = get_batch(split)
            _, loss = model(x, y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)

    return losses

@torch.no_grad()
def evaluate_perplexity(
    model,
    data: torch.Tensor,
    device: torch.device,
    context_length: int = CONTEXT_LENGTH,
    stride: int | None = None
):
    """
    More stable PPL: Slide windows over the full split
    """
    model.eval()
    if stride is None:
        stride = context_length
    
    total_loss: float = 0.0
    n_tokens: int = 0

    # if the length of the data is less than the context length, meaning the data is too short to evaluate perplexity
    if len(data) <= context_length:
        return None
    
    for start in range(0, len(data) - context_length, stride):
        x = data[start:start + context_length].unsqueeze(0).to(device)
        y = data[start + 1 : start + context_length + 1].to(device)
        _, loss = model(x, y)
        n = context_length
        # total loss is the sum of the loss of each token
        total_loss += loss.item() * n
        n_tokens += n
    
    if n_tokens == 0:
        return None

    return_loss = total_loss / n_tokens
    return {
        "loss": return_loss,
        "n_tokens": n_tokens,
        "ppl": math.exp(return_loss)
    }

@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int = 100,
    sampling: SamplingParams | None = None
):
    model.eval()
    sampling = sampling or SamplingParams()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -CONTEXT_LENGTH:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        new_tokens = sample_next_token(
            logits,
            config = sampling,
            previous_tokens = idx[0].tolist()
        )
        idx = torch.cat((idx, new_tokens), dim = 1)
    
    return idx

# now evaluate after generating tokens

def evaluate_generated_text(
    model,
    encode,
    decode,
    device,
    prompts: EVAL_PROMPTS,
    max_new_tokens: int = 150,
):
    print("===== Evaluating Generated Text =====\n")

    for prompt in prompts:
        tokens = encode(prompt)
        if not tokens:
            print(f"Warning: Prompt {prompt} if invalid")
            continue

        context = torch.tensor([tokens], dtype=torch.long, device = device)
        out = generate(model, context, max_new_tokens=max_new_tokens, sampling=SamplingParams)
        text = decode(out[0].tolist())
        continuation = text[len(prompt) :] if text.startswith(prompt) else text

        print(f"Prompt: {prompt!r}")
        print(f"Continuation: {continuation!r}")

# Main function
def main():
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--checkpoint_path", default = CHECKPOINT_PATH)
    parser.add_argument("--data_path", default = DATA_PATH)
    parser.add_argument("--max_new_tokens", type = int | str | None, default = 150)
    parser.add_argument("--n_batches", type = int | str | None, default = 40)
    parser.add_argument("--skip_generation", action = "store_true")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint file not found {args.checkpoint}\n"
            "Train first or pass --checkpoint src/checkpoint.pt"
        )
    
    token = build_tokenizer(args.data)
    encode, decode = token["encode"], token["decode"]
    train_data, val_data = token["train_data"], token["val_data"]