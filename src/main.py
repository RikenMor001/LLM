# nn.ReLu is the activation function
# LLM
# Rotary position embedding
# Multi-head attention
# RMS normalization
# Grouped query attention
# SWIGLU
# Weight tying

import math # square root and scaling attention scores; RMS normalization; SWIGLU
import torch # tensors, GPU acceleration, neural network layers, gradients, and optimization
#from torch._dynamo.polyfills.pytree import tree_unflatten

from torch.fx import Transformer
import torch.nn as nn # linear layers, embeddings, module classes
import torch.nn.functional as F # softmax, silu, dropout, cross entropy
from memory import build_prompt, add_to_memory
from config import BATCH_SIZE, CONTEXT_LENGTH, D_MODEL, N_LAYERS, N_HEADS, N_KV_HEADS, FFN_HIDDEN, DROPOUT, MAX_SEQ_LEN, MAX_STEPS, HEAD_DIM
from models import rmsnorm
from torch.amp import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.checkpoint import checkpoint
from torch.optim.lr_scheduler import CosineAnnealingLR

scaler = GradScaler()

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS device {device}")
else: 
    device = torch.device("cpu") 
    print(f"Using CPU device {device}")

# enables faster matrix multiplication for higher precision
# Given that everything is a matrix multiplication here
torch.set_float32_matmul_precision("high")

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

    ix = torch.randint(
        len(data) - CONTEXT_LENGTH, 
            (BATCH_SIZE, )
    )

    x = torch.stack([
        data[i:i+CONTEXT_LENGTH]
        for i in ix
    ])
    y = torch.stack([
        data[i+1:i+CONTEXT_LENGTH + 1]
        for i in ix
    ])
    return x.to(device), y.to(device)

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

    return torch.sin(angles), torch.cos(angles)

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
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    # two outputs, one you add with cos and other with sin 
    # because we want to keep the position of the tensor x
    output1 = x_even * cos - x_odd* sin
    output2 = x_even * sin + x_odd * cos

    return torch.stack((output1, output2), dim=-1).flatten(-2)

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

# GQA is like a golden cross over for LLM's 

class GQA_Attention(nn.Module):
    def __init__(self, n_kv_heads, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(
            d_model,
            self.head_dim * n_heads,
            bias = False
        )

        self.k_proj = nn.Linear(
            d_model,
            self.head_dim * n_kv_heads,
            bias = False
        )

        self.v_proj = nn.Linear(
            d_model,
            self.head_dim * n_kv_heads,
            bias = False
        )

        self.o_proj = nn.Linear(
            d_model,
            self.head_dim * n_heads,
            bias = False
        )

    def forward(self, x, rope_cos, rope_sin):
            b, seq, _ = x.shape
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            q = q.view(
                b, 
                seq, 
                self.n_heads,
                self.head_dim
            ).transpose(1, 2)

            k = k.view(
                b,
                seq,
                self.n_kv_heads,
                self.head_dim
            ).transpose(1, 2)

            v = v.view(
                b, 
                seq, 
                self.n_kv_heads,
                self.head_dim
            ).transpose(1, 2)

            # apply rope to q and k

            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

            # repeat kv
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=DROPOUT if self.training else 0.0,
                is_causal=True
            )
            out = out.transpose(1, 2).contiguous().view(b, seq, -1)
            out = self.o_proj(out)
            return out

# Feed Forward class

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias = False
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias = False
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias = False
        )

    def forward(self, x):

        return self.w2(
            F.silu(self.w1(x)) * self.w3(x)
        )

# Transformer Block
# add attention, then normalise before FFN, and then feed forward
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = GQA_Attention(
            N_KV_HEADS,
            D_MODEL,
            N_HEADS
        )

        self.feed_forward = FeedForward(
            D_MODEL,
            FFN_HIDDEN
        )

        self.norm1 = rmsnorm.RMSNorm(D_MODEL)
        self.norm2 = rmsnorm.RMSNorm(D_MODEL)
        
    def forward(self, x, rope_cos, rope_sin):

            x = x + self.attention(
                self.norm1(x),
                rope_cos,   
                rope_sin,
            )

            x = x + self.feed_forward(
                self.norm2(x)
            )

            return x

# Full model
class MiniLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(
            len(chars),
            D_MODEL
        )

        self.layers = nn.ModuleList([
            TransformerBlock()
            for _ in range(N_LAYERS)
        ])

        self.norm = rmsnorm.RMSNorm(D_MODEL)

        self.lm_head = nn.Linear(
            D_MODEL,
            len(chars),
            bias=False
        )
        
        self.lm_head.weight = self.token_embedding.weight

        rope_cos, rope_sin = parameters_arrange(
            HEAD_DIM,
            MAX_SEQ_LEN 
        )

        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

    # saves memory but slows the speed for training, because it no longer back propogates, so it doesn't save previous runs, but instead what it does
    # is it runs every layer twice to make it memory efficient. 
    def forward(self, idx, targets = None):
        x = self.token_embedding(idx)

        for layers in self.layers:
            x = checkpoint(
                layers,
                x,
                self.rope_cos,
                self.rope_sin,
                use_reentrant=False
            )

        x = self.norm(x)
        loss = None
        logits = self.lm_head(x)

        if targets is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    label_smoothing=0.1  # Adding label_smoothing so that it gives mixtuer of the ground truth and an uniform distribution.
                ),
        return logits, loss
    
# Validation function
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}

    for split in ["train", "val"]:
        split_losses = []

        for _ in range(20):
            x, y = get_batch_size(split)
            _, loss = model(x, y)
            split_losses.append(loss.item())

        losses[split] = sum(split_losses) / len(split_losses)
    
    model.train()
    print("Losses printed")
    return losses

# training loop
# Generation of text
def generate(model, idx, max_new_tokens = 100):
    model.eval()

    for _ in range(max_new_tokens):

        idx_condition = idx[:, -CONTEXT_LENGTH:]
        logits, _ = model(idx_condition)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probs, 
            num_samples = 1
        )

        idx = torch.cat((idx, next_token), dim = 1)
    return idx

# Creating model, last step

model = MiniLLM().to(device)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 3e-4,
    betas = (0.9, 0.95),
    weight_decay = 0.1 ,
    eps = 1e-8
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max = MAX_STEPS,
    eta_min = 1e-5
)

model.train()
for step in range(MAX_STEPS):
    xb, yb = get_batch_size("train")

    optimizer.zero_grad()
    with autocast(device_type=device.type):
        logits, loss = model(xb, yb)

    # scale, step and update
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if step % 100 == 0:
        losses = estimate_loss()

        print(
            f"Step{step}",
            f"Train Loss: {losses['train']:.4f}"
            f"Val Loss: {losses['val']:.4f}"
        )

        if step % 1000 == 0:
            torch.save(
                model.state_dict(),
                "checkpoint.pt"
            )

print(
    "Total parameters: ",
    sum(p.numel() for p in model.parameters())
)

# Ask Gemini

while True:
    prompt = input("Your question: ")

    if prompt == "exit":
        break

    context = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
        device=device        
    )

    output = generate(
        model,
        context,
        max_new_tokens=200       
    )

    scheduler.step()
    print(
        "LLM",
        decode(output[0].tolist()) # tolist (returns the tensor as a nested list)
    )