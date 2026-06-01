import math
import torch

# make domain perplexity functions

def detect_domain(text: str) -> str:
    # check if domain matches the input or not
    text = text.lower()

    if "risk factors" in text: 
        return "risk"
    elif "revenue" or "income" or "balance sheet" in text:
        return "finance"
    elif "operations" or "management" in text: 
        return "management"
    elif "technology" or "innovation" or "research" in text:
        return "science"
    elif "legal" or "litigation" in text:
        return "law"
    else:
        return "general" or "unknown"