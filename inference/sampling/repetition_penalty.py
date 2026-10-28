"""Repetition penalty."""
import torch
from typing import List, Union


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: Union[List[int], torch.Tensor],
    penalty: float = 1.2
) -> torch.Tensor:
    """
    Apply repetition penalty to logits for generated tokens.
    Divide positive logits and multiply negative logits, as in standard decoding.
    """
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    if penalty == 1.0:
        return logits

    logits = logits.clone()
    was_1d = logits.dim() == 1
    if was_1d:
        logits = logits.unsqueeze(0)
    if isinstance(generated_tokens, torch.Tensor) and generated_tokens.dim() > 1:
        token_rows = generated_tokens.reshape(generated_tokens.size(0), -1).tolist()
    elif isinstance(generated_tokens, (list, tuple)) and generated_tokens and isinstance(generated_tokens[0], (list, tuple)):
        token_rows = [list(row) for row in generated_tokens]
    else:
        shared_tokens = generated_tokens.tolist() if isinstance(generated_tokens, torch.Tensor) else list(generated_tokens or [])
        token_rows = [shared_tokens] * logits.size(0)

    for row_idx, tokens in enumerate(token_rows):
        row = logits[row_idx] if logits.dim() > 1 and len(token_rows) == logits.size(0) else logits
        for token_id in set(tokens):
            if 0 <= int(token_id) < logits.size(-1):
                token_logits = row[..., int(token_id)]
                row[..., int(token_id)] = torch.where(token_logits < 0, token_logits * penalty, token_logits / penalty)

    return logits.squeeze(0) if was_1d else logits
