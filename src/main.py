# nn.ReLu is the activation function
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


# Normalizing the input before passing it 
# to make sure it doesn't go out of bounds

# instead of mean + variance, I use the RMS Normalization technique
# RMSNORM
class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(
            x.pow(2).mean(dim =-1, keepdim = True)
            + self.eps
        )
        x = (x / rms) * self.weight
        return x

# After Loading data, I batched the data and normalized it
# using the RMS Normalization technique, next is 
# ROPE (Rotary Position Embedding)

def parameters_arrange(head_dim, max_seq_len, base = 10000.0):
    freqs = 1.0 / (
        base ** (
            torch.arange(0, head_dim, 2).float() / head_dim
        )
    )

    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)

    return torch.cos(angles), torch.sin(angles)

# turn it with a certain angle to make sure you position 
# it correctly

def apply_rope(x, cos, sin):
    seq_len = x.shape[2] # sequence length keeping it as 2D tensor

    # then I give angles those are cos and sin to the tensor x
    # why do we need pytorch? because tensor works with pytorch

    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0).to(x.device)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0).to(x.device)

    # giving the tensor x position
    # Keeping all the previous dimensions unchanged 
    # only changing the last dimension
    x = x[..., ::2]
    y = x[..., 1::2]

    # two outputs, one you add with cos and other with sin 
    # because we want to keep the position of the tensor x
    output1 = x * cos - y * sin
    output2 = x * sin + y * cos

    return torch.stack(output1, output2, dim=-1).flatten(-2)

# Repeat KV, k = key, v = value
def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x

    b, n_kv, seq, hd = x.shape

    return (
        x[:, :, None, :, :] # just change the middle dimension
        # everything else remain the same, the first 2 and last 2
        .expand(b, n_kv, n_rep, seq, hd)
        .reshape(b, n_kv * n_rep, seq, hd)
    )

# GQA attention
# GQA stands for grouped query attention where we use multiple
# queries but solved by 2 keys and 2 values, makes it more 
# efficient and optimum

class GQA_Attention(nn.Module):
    # __init__ is used to make a constructor
    def __init__(self, n_kv_heads, d_model, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_model = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        # query projection
        self.q_proj = nn.Linear(
            d_model,
            n_heads * self.d_model,
            bias = False
        )

        # key projection
        self.k_proj = nn.Linear(
            d_model,
            n_heads * self.d_model,
            bias = False
        )

        # value projection
        self.v_proj = nn.Linear(
            d_model,
            n_heads * self.d_model,
            bias = False
        )

        # output projection
        self.o_proj = nn.Linear(
            d_model,
            n_heads * self.d_model,
            bias = False
        )

        def forward(self, x, x_cos, x_sin):
            b, seq, _ = x.shape

            q = self.q_proj(x)
            v = self.v_proj(x)
            k = self.k_proj(x)

            q = q.view(
                b,
                seq,
                self.n_heads,
                self.head_dim
            ).transpose(1, 2)

            k = k.view(
                b,
                seq,
                self.head_dim,
                self.n_kv_heads
            ).transpose(1, 2)

            v = v.view(
                b, 
                seq,
                self.head_dim,
                self.n_kv_heads
            ).transpose(1, 2)

        q = apply_rope(q, x_cos, x_sin)
        k = apply_rope(k, x_sin, x_cos)

        v = repeat_kv(v, self.n_rep)
        k = repeat_kv(k, self.n_rep)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch(q @ k.transpose(-2, -1)) * scale

        mask = torch.triu(
            torch.ones(seq, seq, device = x.device),
            diagonal = 1
        ).bool()

        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim = -1)
        weights = F.dropout(
            weights,
            p=DROPOUT,
            training = self.training
        )

        out = weights @ v