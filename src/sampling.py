# to define a common class for all hyperparameters
from dataclasses import dataclass 
import token
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

    for token_id in recent:
        score = logits[:, token_id] # get only the column associated with the token_id
        adjusted = torch.where(
            score > 0,
            score / penalty,
            score * penalty,
        )
        logits[:, token_id] = adjusted
    
    return logits

def apply_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        return logits
    return logits / temperature