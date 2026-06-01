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

def calculate_perplexity(model, max_token_length: 512, texts, tokenizer):
    model.eval()

    for text in texts:
        domain = detect_domain(text)

        inputs = tokenizer(
            text,
            return_tensors = "pt",
            max_length = max_token_length,
            truncation = True
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels = inputs.input_ids)
        
        loss = outputs.loss.item()
        tokens = inputs.input_ids.shape[1]

        # not in domain_stats mean, the found context/domain was not 
        # available which becomes the reason of why there is a loss
        if domain not in domain_stats:
            domain_stats[domain] = {"loss": 0.0, "tokens": 0}

        domain_stats[domain]["loss"] += loss * tokens
        domain_stats[domain]["tokens"] += tokens

    results = {}

    for domain, stats in domain_stats.items():
        results[domain] = stats["loss"] / stats["tokens"]

    return results