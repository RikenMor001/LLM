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

# temperature means the variance of the distribution, higher temperature means more variance
def apply_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        return logits
    return logits / temperature

# top_k means a filter, so I choose the top most quantity of k and the ditribution shouldnt be more than that

def apply_top_k(
    logits: torch.Tensor,
    top_k: int,
    min_tokens_to_keep: int
) -> torch.Tensor:
    k = min(max(min_tokens_to_keep, top_k), logits.size(-1))
    values, _ = torch.topk(logits, k, dim=-1)
    cutoff = values[:, [-1]]
    return logits.masked_fill(logits < cutoff, float('-inf'))

# top_p is to make the sampling process more adaptive with respect to 
# the context of the prompt passed. It does a cumulative sum upto the 
# max set value of p.

def apply_top_p(
    logits: torch.Tensor,
    top_p: float,
    min_tokens_to_keep: int = 1
)-> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    calculated_top_p = (cumulative_probs - sorted_probs) > top_p
    if min_tokens_to_keep > 0:
        calculated_top_p[:, :min_tokens_to_keep] = False

    sorted_logits = sorted_logits.masked_fill(calculated_top_p, float("inf"))
    return sorted_logits.scatter(1, sorted_idx, sorted_logits)