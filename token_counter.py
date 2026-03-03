def estimate_tokens(text: str) -> int:
    """Approximate token count. ~4 chars per token for English text."""
    return len(text) // 4


def fits_in_context(text: str, budget: int) -> bool:
    """Check if text fits within a token budget."""
    return estimate_tokens(text) <= budget


def truncate_to_budget(text: str, budget: int) -> str:
    """Truncate text to fit within a token budget."""
    if fits_in_context(text, budget):
        return text
    max_chars = budget * 4
    return text[:max_chars] + "\n... (truncated to fit context window)"
