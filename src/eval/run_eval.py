import torch

# tells python to save text as strings first instead of 
# evaluating them immediately
from __future__ import annotations

import math
import os
import argparse
 
def build_tokenizer(data_path: str = "input.txt"):
    with open(data_path, "r", encoding="utf-8") as f: # opens the file in read mode with utf-8 encoding
        text = f.read() # reads the entire file into a string