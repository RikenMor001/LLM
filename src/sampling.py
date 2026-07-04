import torch


def sample_next_token(logits, temperature=0.8, top_k=40, top_p=0.9):
    logits = logits / max(temperature, 1e-8)

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        mask = cumulative - probs > top_p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
