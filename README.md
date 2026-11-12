# LLM Cost Router

Most LLM apps send every request to their most capable model, even when
the question is "what's 72°F in Celsius?" That's a lot of money spent on
prompts a model a fraction of the price would answer just as well.

This project sits in front of the API and decides, per request, whether a
prompt actually needs the expensive model. Simple prompts go to **Gemini
3.1 Flash-Lite**, harder ones go to **Gemini 3.5 Flash**, and every query
is logged with what it cost and what it *would* have cost if everything
had gone to the expensive model.

## Results

On a held-out set of 30 prompts the classifier never saw during training:

| Metric | Result |
|---|---|
| Cost vs. always using the expensive model | **63% lower** |
| Routing accuracy | **93%** (28/30) |
| Always-cheap baseline accuracy | 83% (25/30) |

The most interesting finding came from reading the actual answers. Of the
prompts I originally labeled "complex" that got routed to the cheap model,
**the cheap model handled 85% of them well**, including a heapsort
stability proof, a thread-safe LRU cache in Java, and a probability
problem. It only fell short on two: it misstated what proving P = NP would
require, and it described an outdated TLS key exchange. That changed how
I define "complex": by whether the cheap model's answer holds up, not by
how hard the question sounds.

## How it works

```
prompt -> ComplexityClassifier -> Router -> llm_clients -> response
              |                      |             |
       TF-IDF + hand-built     threshold +     Gemini (or OpenAI,
       features -> logistic    code override   or mock mode)
       regression
                                     |
                       cost calc -> SQLite -> /stats API + Streamlit dashboard
```

1. **Classify.** The prompt is turned into features: a TF-IDF vector
   (words and two-word phrases) plus a few hand-built signals: length,
   word count, whether it contains code, reasoning phrases like "step by
   step" or "compare", math symbols, and sentence count. A logistic
   regression model turns those into P(complex).
2. **Route.** If P(complex) ≥ 0.5 the prompt goes to the expensive model.
   Prompts containing real code (a code fence, a function or class
   definition, an import line, SQL) always go to the expensive model.
3. **Call and log.** The router calls the chosen model, prices the
   request using the token counts the API returns, and logs it to SQLite
   next to the cost of sending the same tokens to the expensive model.
4. **Show.** A Streamlit dashboard plots cumulative actual vs. baseline
   spend, the cheap/expensive split, and recent queries.

## Files

| File | What it does |
|---|---|
| `complexity_classifier.py` | Feature extraction and the `ComplexityClassifier` class |
| `train_classifier.py` | Train/test split, fits TF-IDF + scaler + logistic regression, saves `classifier.joblib` |
| `router.py` | Routing decision, model call, cost calculation, logging |
| `llm_clients.py` | Calls Gemini through its OpenAI-compatible endpoint (or OpenAI, or mock mode with no key) |
| `pricing_config.py` | Loads `.env`, picks the provider, holds per-model prices |
| `db.py` | SQLite logging and stats queries |
| `main.py` | FastAPI app: `POST /query`, `GET /stats`, `GET /health` |
| `dashboard.py` | Streamlit dashboard |
| `send_prompts.py` | Sends a CSV of prompts to the API, scores routing against labels, saves every answer |
| `data/training_prompts.csv` | 52 labeled training prompts |
| `data/test_prompts.csv` | 30 held-out test prompts |
| `data/build_dataset.py` | Generates synthetic template prompts for a quick demo run |

## Running it

```bash
pip install -r requirements.txt

# Add your key (a free Gemini key works: https://aistudio.google.com/apikey)
cp .env.example .env        # then paste your key into .env

python train_classifier.py  # train the classifier
uvicorn main:app --reload   # start the API at http://127.0.0.1:8000
streamlit run dashboard.py  # in a second terminal
```

Try a request at **http://127.0.0.1:8000/docs**, or with curl:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

To run the whole test set and see which prompts get misrouted:

```bash
python send_prompts.py data/test_prompts.csv
```

Every answer is saved to `results/` so you can read what each model
actually said.

With no API key the app runs in **mock mode** with fake responses, so you
can try the pipeline for free. Mock rows are flagged in the database and
hidden on the dashboard by default.

## How cost is measured

Each query is priced at the provider's paid-tier rates using the real
token counts returned by the API. The baseline prices the same tokens at
the expensive model's rate. Larger models usually write longer answers,
so this baseline understates what the expensive model would really cost,
which makes the 63% figure a conservative one.

## Limitations

- The training and test sets are small (52 and 30 prompts), so accuracy
  numbers move noticeably with each prompt.
- Training labels were based on how hard a prompt sounds; test labels are
  based on answer quality. Relabeling the training set the same way is
  the obvious next step.
- If a real API call fails (bad key, rate limit), the request falls back
  to a mock reply and is logged as mocked so it never counts toward the
  real numbers.

## What I'd do next

- Relabel training data by answer quality and grow both datasets.
- Use `class_weight="balanced"` to offset the simple/complex imbalance.
- Try sentence embeddings instead of TF-IDF.
- Add a feedback endpoint so misrouted prompts flow back into training.
- Cache repeated or near-duplicate prompts.
