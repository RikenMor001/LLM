# Normalizing the input before passing it 
# to make sure it doesn't go out of bounds

# instead of mean + variance, I use the RMS Normalization technique
# RMSNORM
import torch
import torch.nn as nn
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