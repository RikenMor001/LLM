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
