"""
Combines the complexity classifier's probability with a couple of hard
override rules to make the final routing decision, then calls the model
and logs cost data.
"""
from complexity_classifier import ComplexityClassifier, CODE_BLOCK_RE
from llm_clients import call_model
from pricing_config import cost_for
import db

COMPLEXITY_THRESHOLD = 0.5

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = ComplexityClassifier.load()
    return _classifier


def decide_route(prompt: str) -> tuple[str, float]:
    """Returns (route, complexity_score)."""
    clf = get_classifier()
    score = clf.predict_proba_complex(prompt)

    # Hard override: anything containing a code block is treated as
    # complex regardless of the classifier's score, since routing actual
    # code-debugging tasks to the cheap model tends to produce bad answers.
    if CODE_BLOCK_RE.search(prompt):
        return "expensive", max(score, 0.9)

    route = "expensive" if score >= COMPLEXITY_THRESHOLD else "cheap"
    return route, score


def handle_query(prompt: str) -> dict:
    route, score = decide_route(prompt)
    result = call_model(prompt, route)

    actual_cost = cost_for(route, result["input_tokens"], result["output_tokens"])
    baseline_cost = cost_for("expensive", result["input_tokens"], result["output_tokens"])

    db.log_query(
        prompt=prompt,
        complexity_score=score,
        route=route,
        model_used=result["model"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        actual_cost=actual_cost,
        baseline_cost=baseline_cost,
        mocked=result["mocked"],
    )

    return {
        "response": result["text"],
        "route": route,
        "model_used": result["model"],
        "complexity_score": round(score, 3),
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "actual_cost": round(actual_cost, 6),
        "baseline_cost": round(baseline_cost, 6),
        "saved": round(baseline_cost - actual_cost, 6),
        "mocked": result["mocked"],
    }
