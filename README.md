# LLM Cost Router

A routing layer that sits in front of LLM APIs, classifies each incoming
prompt's *complexity*, and sends simple prompts to a cheap model while
reserving an expensive model for prompts that actually need it — then
tracks the dollar savings vs. a baseline of always calling the expensive
model.

## Why this project

Sending a request to a frontier model when a cheaper model could have
handled it just as well is wasted spend. This project measures that waste
and routes around it automatically.

## Architecture

```
prompt -> ComplexityClassifier -> Router -> llm_clients -> response
                |                    |            |
          (TF-IDF + hand       (threshold +   (OpenAI, or mock
           features ->          code-prompt     mode with no key)
           LogisticRegression)  override)
                                     |
                              cost calc -> SQLite -> FastAPI /stats + Streamlit dashboard
```

- **data/build_dataset.py**: generates a labeled set of simple vs. complex
  example prompts into `data/training_prompts.csv`.
- **complexity_classifier.py**: turns a prompt into features: a TF-IDF
  vector of the text (words and two-word phrases) plus hand-engineered
  signals (character length, word count, whether it contains code,
  phrases like "step by step" / "compare" / "debug", math symbols,
  sentence count). Also holds the `ComplexityClassifier` class that loads
  the trained model and returns P(complex) for a prompt.
- **train_classifier.py**: splits the data into train/test, fits the
  TF-IDF vectorizer, a `StandardScaler` for the hand features, and a
  `LogisticRegression`, prints a classification report on the test set,
  and saves everything to `classifier.joblib`.
- **router.py**: routes to the expensive model if P(complex) >= 0.5,
  with one hard override: prompts containing code always go to the
  expensive model. Then it calls the model, computes cost, and logs it.
- **llm_clients.py**: wraps the model call. With `GEMINI_API_KEY` in `.env`
  it calls Gemini 3.1 Flash-Lite (cheap) or Gemini 3.5 Flash (expensive)
  through Gemini's OpenAI-compatible endpoint. With `OPENAI_API_KEY`
  instead it calls gpt-4o-mini / gpt-4o. With no key it runs in
  **mock mode** (fake responses and token counts). Mock rows are flagged
  in the database.
- **pricing_config.py**: loads `.env`, picks the provider, and holds
  per-model $/1K-token rates (paid-tier list prices) and the cost formula.
- **db.py**: SQLite logging of every query: complexity score, route,
  model, tokens, actual cost, baseline cost, and whether it was mocked.
- **main.py**: FastAPI app exposing `POST /query`, `GET /stats`
  (`?real_only=true` excludes mock rows), and `GET /health`.
- **dashboard.py**: Streamlit dashboard reading from the same SQLite DB:
  summary metrics, cumulative actual-vs-baseline cost over time, route
  distribution, and a recent-query log. Mock rows are hidden by default.

## Running it

```bash
cd llm-router
pip install -r requirements.txt

# 1. Train the classifier (generates the dataset first if missing)
python train_classifier.py

# 2. Start the API (mock mode unless OPENAI_API_KEY is set)
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

To use real models, copy `.env.example` to `.env` and paste a key:
```
GEMINI_API_KEY=your-key-here
```
Then restart uvicorn. `.env` is in `.gitignore`, so the key never gets committed.
If you previously ran in mock mode, either delete `router_log.db` or keep
the dashboard's "Real API calls only" toggle on, so fake rows don't mix
into your numbers.

## How cost is measured

Costs are computed at each provider's **paid-tier list prices**. On
Gemini's free tier the actual bill is $0, so the numbers show what the
same traffic would cost at paid rates.

For each query, `actual_cost` uses the routed model's price and
`baseline_cost` uses the expensive model's price for the **same token
counts**. That assumes the expensive model would have produced an answer
of the same length. In practice larger models often write longer answers,
so the reported savings are likely conservative.

## Known limitations

- The bundled training data is template-generated, so test accuracy on it
  is not meaningful. Real accuracy needs hand-labeled prompts.
- `estimate_tokens` (words x 1.3) is only used in mock mode; real mode
  uses the token counts the API returns.
- If a real API call fails (bad key, free-tier rate limit), the request
  falls back to a mock reply and is logged with `mocked = 1`.
- Gemini 3.x Flash models "think" before answering. Whether thinking tokens
  show up in the reported output token count via the OpenAI-compatible
  endpoint should be checked against Google's usage dashboard.

## Possible extensions

- Semantic caching of repeated/similar queries.
- A feedback endpoint so mis-routed queries get added back to the
  training set.
- Sentence-embedding features instead of TF-IDF.
- Exact token counting with `tiktoken`.
