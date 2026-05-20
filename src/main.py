# LLM
# Rotary position embedding
# Multi-head attention
# Grouped query attention
# RMS normalization
# SWIGLU
# Weight tying

import math # square root and scaling attention scores; RMS normalization; SWIGLU
import torch # tensors, GPU acceleration, neural network layers, gradients, and optimization
import torch.nn as nn # linear layers, embeddings, module classes
import torch.nn.functional as F # softmax, silu, dropout, cross entropy

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS device {device}")
else: 
    device = torch.device("cpu") 
    print(f"Using CPU device {device}")

# enables faster matrix multiplication for higher precision
# Given that everything is a matrix multiplication here
torch.set_float32_matmul_precision("high")

# config

BATCH_SIZE = 32
CONTEXT_LENGTH = 128
MAX_STEPS = 3000
EVAL_INTERVAL = 200
LEARNING_RATE = 3e-4

D_MODEL = 256
N_LAYERS = 4
N_HEADS = 8
N_KV_HEADS = 2 # This is grouped query attention
HEAD_DIM = D_MODEL // N_HEADS
FFN_HIDDEN = 680 # FFN = Feed Forward Network
DROPOUT = 0.2
MAX_SEQ_LEN = 256 # This is the maximum sequence length for the model

# LOADING DATA
with open("input.txt", "r", encoding = "utf-8") as f:
    text = f.read()
    print(f"Loaded {len(text)} characters in the LLM training data") # returns the number of items in the container

# TOKENZATION
chars = sorted(list(set(text)))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}

def encode(string):
    return [char_to_index[c] for c in string]

def decode(tokens):
    return "".join([index_to_char[i] for i in tokens])

data = torch.tensor(encode(text), dtype = torch.long)
split = int(0.9 * len(data))

train_data = data[:split]
val_data = data[split:]

# BATCHING

def get_batch_size(split):
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - CONTEXT_LENGTH, (BATCH_SIZE))

    x = torch.stack([
        data[i:i+CONTEXT_LENGTH]
        for i in ix
    ])
    y = torch.stack([
        data[i+1:i+CONTEXT_LENGTH + 1]
        for i in ix
    ])
    return x.to(device), y.to(device)

# RMSNORM
# Normalizing the input before passing it 
# to make sure it doesn't go out of bounds
class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(
            x.pow(2).mean(dim = -1, keepdim = True)
            + self.eps
        )

        return (x / rms) * self.weight