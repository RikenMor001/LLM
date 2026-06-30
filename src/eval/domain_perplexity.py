import math

import torch

DOMAIN_EVAL_TEXTS = [
    (
        "The afternoon sun filtered through the dense canopy of the old oak tree, "
        "casting dancing shadows on the worn wooden picnic table below."
    ),
    (
        "Revenue increased year over year and net income rose on the balance sheet "
        "as operating margins improved across all segments."
    ),
    (
        "Item 1A risk factors include market volatility, supply chain disruption, "
        "and regulatory changes that may adversely affect our business."
    ),
    (
        "Management discussion of operations highlights strategic priorities, "
        "resource allocation, and oversight of day to day business activities."
    ),
    (
        "Technology innovation and research programs drive next generation products "
        "and advance our scientific capabilities."
    ),
    (
        "Legal proceedings and litigation may materially affect our financial "
        "condition, results of operations, and cash flows."
    ),
]


def detect_domain(text: str) -> str:
    text = text.lower()

    if "risk factors" in text:
        return "risk"
    if any(k in text for k in ("revenue", "income", "balance sheet")):
        return "finance"
    if any(k in text for k in ("technology", "innovation", "research")):
        return "science"
    if any(k in text for k in ("legal", "litigation")):
        return "law"
    if any(k in text for k in ("operations", "management")):
        return "management"
    return "general"


def calculate_domain_perplexity(model, texts, encode_fn, device, max_length=512):
    model.eval()
    domain_stats = {}

    for text in texts:
        domain = detect_domain(text)
        tokens = encode_fn(text)[:max_length]
        if len(tokens) < 2:
            continue

        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        targets = idx.clone()

        with torch.no_grad():
            _, loss = model(idx, targets)

        n_tokens = idx.shape[1]
        if domain not in domain_stats:
            domain_stats[domain] = {"loss": 0.0, "tokens": 0}
        domain_stats[domain]["loss"] += loss.item() * n_tokens
        domain_stats[domain]["tokens"] += n_tokens

    results = {}
    for domain, stats in domain_stats.items():
        avg_loss = stats["loss"] / stats["tokens"]
        results[domain] = math.exp(avg_loss)

    return results


def log_domain_perplexity(results: dict):
    if not results:
        return

    print("Domain perplexity:")
    for domain, ppl in sorted(results.items()):
        print(f"  {domain}: {ppl:.2f}")
