"""
Classifies a prompt as "simple" (cheap model is fine) or "complex"
(needs the expensive model).

Deliberately built on TF-IDF + hand-engineered features + logistic
regression rather than an embedding model or an LLM call itself:
- it's cheap and fast enough to run on every single request
- it's fully explainable — you can point to exactly which words/features
  drove a decision, which matters both for debugging and for defending
  the design in an interview
"""
import re
import joblib
import numpy as np
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

MODEL_PATH = Path(__file__).parent / "classifier.joblib"

CODE_BLOCK_RE = re.compile(r"```|def |class |import |SELECT |function\(")
STEP_WORDS_RE = re.compile(
    r"\bstep by step\b|\bwalk me through\b|\bexplain how\b|\bcompare\b|"
    r"\bdesign\b|\bderive\b|\brefactor\b|\boptimi[sz]e\b|\bdebug\b",
    re.IGNORECASE,
)
MATH_SYMBOLS_RE = re.compile(r"[=<>%^]|O\(")


def hand_features(prompts):
    """Returns an (n_samples, k) array of numeric features per prompt."""
    feats = []
    for p in prompts:
        length = len(p)
        n_words = len(p.split())
        has_code = 1.0 if CODE_BLOCK_RE.search(p) else 0.0
        has_step_words = 1.0 if STEP_WORDS_RE.search(p) else 0.0
        has_math = 1.0 if MATH_SYMBOLS_RE.search(p) else 0.0
        n_sentences = max(1, p.count(".") + p.count("?") + p.count("!"))
        feats.append([length, n_words, has_code, has_step_words, has_math, n_sentences])
    return np.array(feats, dtype=float)


class ComplexityClassifier:
    def __init__(self, vectorizer=None, scaler=None, model=None):
        self.vectorizer = vectorizer
        self.scaler = scaler
        self.model = model

    def _build_matrix(self, prompts):
        tfidf = self.vectorizer.transform(prompts)
        hand = self.scaler.transform(hand_features(prompts))
        return hstack([tfidf, csr_matrix(hand)])

    def predict_proba_complex(self, prompt: str) -> float:
        """Returns P(complex) in [0, 1] for a single prompt."""
        X = self._build_matrix([prompt])
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        return float(proba[classes.index("complex")])

    def save(self):
        joblib.dump(
            {"vectorizer": self.vectorizer, "scaler": self.scaler, "model": self.model},
            MODEL_PATH,
        )

    @classmethod
    def load(cls):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "No trained classifier found. Run `python train_classifier.py` first."
            )
        blob = joblib.load(MODEL_PATH)
        return cls(vectorizer=blob["vectorizer"], scaler=blob["scaler"], model=blob["model"])
