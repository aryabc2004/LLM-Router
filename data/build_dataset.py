"""
PLACEHOLDER DATA GENERATOR.

Builds a small synthetic, template-generated set of labeled prompts so the
pipeline can be trained and run end to end. This is NOT real data: the
templates make the two classes easy to separate, so any accuracy measured
on it is meaningless. Replace data/training_prompts.csv with real,
hand-labeled prompts before reporting any numbers.

Usage:
    python data/build_dataset.py
"""
import csv
import random
from pathlib import Path

OUT_PATH = Path(__file__).parent / "training_prompts.csv"
random.seed(42)

# ---- simple: short factual / lookup / rewrite tasks ----
SIMPLE_TEMPLATES = [
    "What is the capital of {country}?",
    "Translate '{phrase}' into {language}.",
    "How many {unit_pair}?",
    "Who wrote {book}?",
    "Define the word '{word}'.",
    "What year did {event} happen?",
    "Give me a synonym for '{word}'.",
    "Fix the spelling in this sentence: '{typo}'",
]
SIMPLE_FILL = {
    "country": ["France", "Japan", "Brazil", "Kenya", "Canada", "Italy"],
    "phrase": ["good morning", "thank you", "where is the library", "see you later"],
    "language": ["Spanish", "French", "German", "Hindi"],
    "unit_pair": ["ounces are in a pound", "inches are in a foot",
                  "minutes are in an hour", "centimeters are in a meter"],
    "book": ["Hamlet", "Pride and Prejudice", "1984", "The Great Gatsby"],
    "word": ["happy", "ephemeral", "quick", "verbose", "resilient"],
    "event": ["the moon landing", "the fall of the Berlin Wall", "the first iPhone release"],
    "typo": ["I recieved the pakage", "Thier dog is realy cute", "Its definately raining"],
}

# ---- complex: multi-step reasoning / code / design / comparison ----
COMPLEX_TEMPLATES = [
    "Explain step by step how {system} works, including the main failure modes.",
    "Write a Python function that implements {algorithm} and analyze its time complexity.",
    "Compare {option_a} and {option_b} for {use_case}, and recommend one with tradeoffs.",
    "Design a {product} that can handle {scale}. Walk me through the architecture.",
    "Debug this code and explain the fix:\n```python\n{snippet}\n```",
    "Derive {math_result} from first principles.",
    "Refactor this {language} code to be more maintainable and explain each change.",
    "Optimize a {db_task} and explain why your approach is faster.",
]
COMPLEX_FILL = {
    "system": ["TCP congestion control", "garbage collection in Java", "a transformer's attention",
               "DNS resolution", "Raft consensus"],
    "algorithm": ["Dijkstra's algorithm", "an LRU cache", "merge sort", "A* search", "a trie"],
    "option_a": ["PostgreSQL", "REST", "React", "Kafka"],
    "option_b": ["MongoDB", "GraphQL", "Vue", "RabbitMQ"],
    "use_case": ["a social media backend", "a real-time chat app", "an analytics pipeline"],
    "product": ["URL shortener", "rate limiter", "news feed", "distributed cache"],
    "scale": ["10 million daily users", "100k requests per second", "global traffic"],
    "snippet": ["def avg(xs):\n    return sum(xs) / len(xs)",
                "def fib(n):\n    return fib(n-1) + fib(n-2)",
                "for i in range(len(items)):\n    items.remove(items[i])"],
    "math_result": ["the gradient of the logistic regression loss", "the formula for variance",
                    "the closed-form solution to linear regression"],
    "language": ["JavaScript", "Java", "C++"],
    "db_task": ["SQL query that joins three large tables", "slow nested-loop lookup"],
}

N_PER_CLASS = 70


def fill(template, values):
    out = template
    for key, options in values.items():
        placeholder = "{" + key + "}"
        if placeholder in out:
            out = out.replace(placeholder, random.choice(options))
    return out


def generate(templates, values, n):
    prompts = set()
    attempts = 0
    while len(prompts) < n and attempts < n * 50:
        prompts.add(fill(random.choice(templates), values))
        attempts += 1
    return sorted(prompts)


def main():
    simple = generate(SIMPLE_TEMPLATES, SIMPLE_FILL, N_PER_CLASS)
    complex_ = generate(COMPLEX_TEMPLATES, COMPLEX_FILL, N_PER_CLASS)
    rows = [(p, "simple") for p in simple] + [(p, "complex") for p in complex_]
    random.shuffle(rows)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} PLACEHOLDER prompts "
          f"({len(simple)} simple, {len(complex_)} complex) to {OUT_PATH}")


if __name__ == "__main__":
    main()
