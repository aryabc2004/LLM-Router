"""
Per-model pricing, in dollars per 1,000 tokens.

These are placeholder / illustrative values. Prices change often and
differ by provider, so before using this for a real demo or resume
screenshot, replace them with current published pricing from the
provider's own pricing page.
"""

# (input_price_per_1k, output_price_per_1k)
PRICING = {
    "cheap": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "input_per_1k": 0.00015,
        "output_per_1k": 0.00060,
    },
    "expensive": {
        "provider": "openai",
        "model": "gpt-4o",
        "input_per_1k": 0.0025,
        "output_per_1k": 0.0100,
    },
}


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate (no external tokenizer dependency / network
    call). English text averages roughly 1.3 tokens per whitespace-
    separated word - good enough for relative cost comparisons, not
    exact billing.
    """
    words = len(text.split())
    return max(1, int(words * 1.3))


def cost_for(route: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING[route]
    return (
        input_tokens / 1000 * rates["input_per_1k"]
        + output_tokens / 1000 * rates["output_per_1k"]
    )
