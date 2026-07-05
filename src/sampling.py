# to define a common class for all hyperparameters
from dataclasses import dataclass 
from typing import Optional, Sequence
import torch

@dataclass
class SamplingParams:
    temperatture: float = 0.8
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.9
    repetition_penalty: float = 1.15
    repetition_window: int = 64

def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: Sequence[int],
    penalty: float,
    window: int,
) -> torch.Tensor:
    if penalty == 1.0 or not token_ids:
        return logits

    logits = logits.clone()
    recent = list(dict.fromkeys(token_ids[-window:]))