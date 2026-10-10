"""
Wraps the actual model call for each route ("cheap" / "expensive").

If OPENAI_API_KEY is set, cheap/expensive both call OpenAI models.
If no key is set, falls back to MOCK MODE: generates a plausible fake
response and token counts, so the whole pipeline (router, cost tracker,
dashboard) works end-to-end with zero API cost. This is what lets you
demo the project without paying for API calls.
"""
import os
import random
from pricing_config import PRICING, estimate_tokens

_openai_client = None
if os.environ.get("OPENAI_API_KEY"):
    try:
        from openai import OpenAI
        _openai_client = OpenAI()
    except ImportError:
        _openai_client = None


def _mock_response(prompt: str, route: str) -> tuple[str, int, int]:
    """Simulates a response + token counts, scaled by route and prompt length."""
    input_tokens = estimate_tokens(prompt)
    if route == "cheap":
        reply = f"[mock cheap-model reply] Short answer for: {prompt[:60]}..."
        output_tokens = random.randint(15, 60)
    else:
        reply = (
            f"[mock expensive-model reply] Detailed, multi-step answer for: "
            f"{prompt[:60]}... (would include reasoning, code, and caveats)"
        )
        output_tokens = random.randint(150, 400)
    return reply, input_tokens, output_tokens


def call_model(prompt: str, route: str) -> dict:
    """
    Returns {"text": str, "input_tokens": int, "output_tokens": int,
    "model": str, "mocked": bool}
    """
    model_name = PRICING[route]["model"]

    if _openai_client is not None:
        try:
            resp = _openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = resp.usage
            return {
                "text": resp.choices[0].message.content,
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "model": model_name,
                "mocked": False,
            }
        except Exception as e:
            # Fall through to mock mode if the real call fails, so the
            # demo never breaks mid-presentation.
            text, in_tok, out_tok = _mock_response(prompt, route)
            return {
                "text": f"[API call failed ({e}), showing mock reply] " + text,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "model": model_name,
                "mocked": True,
            }

    text, in_tok, out_tok = _mock_response(prompt, route)
    return {
        "text": text,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "model": model_name,
        "mocked": True,
    }
