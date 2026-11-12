"""
Per-model pricing, in dollars per 1,000 tokens.

Which provider is used depends on which API key is set in .env:
GEMINI_API_KEY -> Gemini, otherwise OPENAI_API_KEY -> OpenAI,
otherwise mock mode (uses the Gemini table for fake costs).

Prices are the PAID-tier list prices (checked Nov 2026). On Gemini's free
tier you are billed $0, so the dashboard shows what the same traffic would
cost at paid rates. Re-check the provider pricing pages before publishing
numbers, and if you change a model name, update its prices too.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads key=value lines from .env into environment variables

PRICING_BY_PROVIDER = {
    "gemini": {
        "cheap": {
            "model": "gemini-3.1-flash-lite",
            "input_per_1k": 0.00025,   # $0.25 / 1M
            "output_per_1k": 0.0015,   # $1.50 / 1M
        },
        "expensive": {
            "model": "gemini-3.5-flash",
            "input_per_1k": 0.0015,    # $1.50 / 1M
            "output_per_1k": 0.009,    # $9.00 / 1M
        },
    },
    "openai": {
        "cheap": {
            "model": "gpt-4o-mini",
            "input_per_1k": 0.00015,   # $0.15 / 1M
            "output_per_1k": 0.00060,  # $0.60 / 1M
        },
        "expensive": {
            "model": "gpt-4o",
            "input_per_1k": 0.0025,    # $2.50 / 1M
            "output_per_1k": 0.0100,   # $10.00 / 1M
        },
    },
}

if os.environ.get("GEMINI_API_KEY"):
    PROVIDER = "gemini"
elif os.environ.get("OPENAI_API_KEY"):
    PROVIDER = "openai"
else:
    PROVIDER = "gemini"  # mock mode; just picks which price table the fake costs use

PRICING = PRICING_BY_PROVIDER[PROVIDER]


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate (no external tokenizer dependency / network
    call). English text averages roughly 1.3 tokens per whitespace-
    separated word - good enough for relative cost comparisons, not
    exact billing. Only used in mock mode.
    """
    words = len(text.split())
    return max(1, int(words * 1.3))


def cost_for(route: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING[route]
    return (
        input_tokens / 1000 * rates["input_per_1k"]
        + output_tokens / 1000 * rates["output_per_1k"]
    )
