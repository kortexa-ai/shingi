"""Synthetic context probes, separate from real-document quality evidence."""
import random


def context_record(repetitions, position, seed):
    rng = random.Random(seed)
    colors = ["amber", "teal", "violet", "silver"]
    target = colors[rng.randrange(len(colors))]
    rng.shuffle(colors)
    marker = f"Audit entry: the access color for room K17 is {target}.\n"
    filler = "Inventory entry: a wooden table, two ceramic cups, and a copper lamp.\n"
    before = round(repetitions * position)
    return {"state": filler * before + marker + filler * (repetitions - before),
            "question": {"type": "choice", "instructions": "According to the audit entry, what is the access color for room K17?",
                         "criteria": {color: None for color in colors}}, "label": target}


def probe_at_tokens(backend, target_tokens, position, seed):
    from .decision import prompt_for
    low, high = 0, target_tokens
    best = None
    while low <= high:
        mid = (low + high) // 2
        row = context_record(mid, position, seed)
        prompt = prompt_for(row["state"], row["question"]["instructions"], list(row["question"]["criteria"].items()))
        try:
            count = backend.infer(prompt, list("ABCD"), tokenize_only=True)["input_tokens"]
        except ValueError as exc:
            if "exceeds context" not in str(exc):
                raise
            high = mid - 1
            continue
        if count <= target_tokens:
            best = row, count
            low = mid + 1
        else:
            high = mid - 1
    if best is None:
        raise ValueError("requested target cannot fit even the minimal prompt")
    return best
