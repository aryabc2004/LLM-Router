# LLM Cost Router

A routing layer that sits in front of LLM APIs, classifies each incoming
prompt's *complexity*, and sends simple prompts to a cheap model while
reserving an expensive model for prompts that actually need it — then
tracks the dollar savings vs. a baseline of always calling the expensive
model.

## Why this project

Every extra request sent to a frontier model that a cheaper model could
have handled just as well is wasted spend. This project quantifies that
waste and eliminates most of it automatically, which is the same "measure
it, then fix it" story that makes infra/LLM-ops projects land well in
interviews.

## Architecture

```
prompt -> ComplexityClassifier -> Router -> LLMClient -> response
                |                    |            |
          (TF-IDF + hand      (threshold    (mock or real
           features -> Logistic  rule)        OpenAI/Anthropic)
           Regression)
                                    |
                              CostTracker -> SQLite -> Streamlit dashboard
```

- **complexity_classifier.py** — turns a prompt into features (TF-IDF
  vector of the text + hand-engineered signals: length, presence of a
  code block, question words like "why"/"explain step by step", math
  symbols) and feeds them into a `LogisticRegression` trained on a
  labeled set of simple vs. complex example prompts (`train_classifier.py`
  generates this dataset and trains the model).
- **router.py** — takes the classifier's complexity probability and
  applies a threshold (plus a couple of hard override rules, e.g. "always
  treat code blocks as complex") to decide: cheap model or expensive
  model.
- **llm_clients.py** — wraps the actual model call. If you set
  `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` it calls the real API; with no
  key set it runs in **mock mode**, simulating a response and token
  counts so the whole pipeline (and the dashboard) works out of the box
  with zero cost.
- **pricing_config.py** — per-model $/1K-token rates. These are
  placeholder example values — update them to current published pricing
  before you screenshot the dashboard for a resume/demo.
- **db.py** — SQLite logging of every query: complexity score, which
  model was used, estimated tokens, actual cost, and what it *would have*
  cost if routed to the expensive model every time.
- **main.py** — FastAPI app exposing `POST /query` and `GET /stats`.
- **dashboard.py** — Streamlit dashboard reading from the same SQLite DB:
  cumulative actual-vs-baseline cost, route distribution, recent query
  log.

## Running it

```bash
cd llm-router
pip install -r requirements.txt

# 1. Train the classifier (fast, no GPU needed, ~1 second)
python train_classifier.py

# 2. Start the API (mock mode by default — no API key needed)
uvicorn main:app --reload

# 3. In a second terminal, start the dashboard
streamlit run dashboard.py
```

Send a test query:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function that implements Dijkstras algorithm and explain the time complexity step by step, comparing it to Bellman-Ford."}'
```

The first should route to the cheap model, the second to the expensive
one. Refresh the dashboard to see the cost split update.

To use real models instead of mock mode:
```bash
export OPENAI_API_KEY=sk-...      # optional
export ANTHROPIC_API_KEY=sk-ant-...  # optional
```

## How to explain this in an interview

- **The problem**: naive LLM apps send every request to the most capable
  (most expensive) model, even for trivial prompts.
- **The approach**: a supervised classifier — not an LLM call itself,
  which would defeat the purpose — decides complexity cheaply and fast
  using TF-IDF + logistic regression, so the routing decision itself
  costs essentially nothing.
- **The measurement**: every query logs what it actually cost vs. what
  the "always use the expensive model" baseline would have cost, so the
  savings number is real and reproducible from your own data, not a
  guess.
- **Where you'd take it further**: caching repeated/similar queries
  (semantic caching), a feedback loop that re-labels misrouted queries
  (e.g. a cheap-model response that the user regenerates), or swapping
  the linear classifier for a distilled small model if TF-IDF features
  stop being discriminative enough.

## Extending it

- Swap TF-IDF for sentence-transformer embeddings if you want higher
  accuracy and are comfortable explaining embeddings in an interview.
- Add a feedback endpoint so mis-routed queries (user says "answer was
  bad") get added back into the training set.
- Add real token counting (`tiktoken`) instead of the whitespace-based
  approximation currently used.
