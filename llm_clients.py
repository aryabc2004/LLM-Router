"""
Wraps the actual model call for each route ("cheap" / "expensive").

Keys are read from .env (see .env.example):
- GEMINI_API_KEY set: calls Gemini through its OpenAI-compatible endpoint,
  so the same `openai` library works with only a different base_url.
- else OPENAI_API_KEY set: calls OpenAI.
- else MOCK MODE: generates a plausible fake response and token counts,
  so the whole pipeline (router, cost tracker, dashboard) works end-to-end
  with zero API cost.
"""
import os
import random
# Importing pricing_config first also runs load_dotenv(), so the keys
# from .env are in os.environ before we check for them below.
from pricing_config import PRICING, PROVIDER, estimate_tokens

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

_openai_client = None
try:
    from openai import OpenAI
    if PROVIDER == "gemini" and os.environ.get("GEMINI_API_KEY"):
        _openai_client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_BASE_URL)
    elif PROVIDER == "openai" and os.environ.get("OPENAI_API_KEY"):
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
