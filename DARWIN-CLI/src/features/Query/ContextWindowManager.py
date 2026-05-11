# src/features/Query/ContextWindowManager.py
"""
Token-aware context trimming utilities.
"""
from __future__ import annotations

import tiktoken

# Encoder for token counting (cl100k_base is a good default for recent models;
# adjust if you're using a different family)
_ENCODER = tiktoken.get_encoding("cl100k_base")

# Fallback character-based estimation for models not in tiktoken
def estimate_tokens(text: str) -> int:
    """Return token count using tiktoken, falling back to char/4."""
    try:
        return len(_ENCODER.encode(text))
    except Exception:
        return len(text) // 4


def trim_contexts(
    contexts: list[str],
    max_tokens: int,
) -> list[str]:
    """
    Keep contexts list until cumulative token count exceeds max_tokens.
    Assumes contexts are sorted by relevance (highest first).
    """
    trimmed = []
    total = 0
    for ctx in contexts:
        tokens = estimate_tokens(ctx)
        if total + tokens <= max_tokens:
            trimmed.append(ctx)
            total += tokens
        else:
            # If the first context already exceeds the budget, truncate it
            if not trimmed:
                # Truncate ctx to fit roughly max_tokens (4 chars/token)
                trimmed.append(ctx[: max_tokens * 4])
            break
    return trimmed


def trim_history(history_summary: str, max_tokens: int) -> str:
    """Truncate the conversation summary to max_tokens."""
    if not history_summary:
        return ""
    tokens = estimate_tokens(history_summary)
    if tokens <= max_tokens:
        return history_summary
    # Simple truncation from the head (keep recent)
    # Better: use token-aware slicing
    return _ENCODER.decode(_ENCODER.encode(history_summary)[-max_tokens:])