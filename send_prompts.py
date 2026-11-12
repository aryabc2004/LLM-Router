"""
Sends every prompt in a CSV to the running API (POST /query) and compares
the route it chose with the expected label.

Usage (with uvicorn already running in another terminal):
    python send_prompts.py                                  # uses data/my_prompts_labeled.csv
    python send_prompts.py data/some_other_file.csv --delay 8

CSV format: prompt,label   (label = simple or complex; can be blank)
The delay between requests keeps you under Gemini's free-tier rate limit.
"""
import argparse
import csv
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

API_URL = "http://127.0.0.1:8000/query"
EXPECTED_ROUTE = {"simple": "cheap", "complex": "expensive"}


def send(prompt: str) -> dict:
    body = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default="data/my_prompts_labeled.csv")
    parser.add_argument("--delay", type=float, default=5.0, help="seconds to wait between requests")
    parser.add_argument("--start", type=int, default=1, help="prompt number to start from (to resume a run)")
    args = parser.parse_args()

    # Every answer is saved here so you can read what the model actually said.
    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / f"responses_{datetime.now():%Y%m%d_%H%M%S}.md"
    out = open(out_path, "w", encoding="utf-8")

    with open(Path(args.csv_path), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Sending {len(rows)} prompts to {API_URL} ({args.delay}s apart)...\n")

    correct = labeled = mocked = 0
    misses = []
    for i, row in enumerate(rows, 1):
        if i < args.start:
            continue
        prompt = row["prompt"]
        label = (row.get("label") or "").strip().lower()
        try:
            r = send(prompt)
        except Exception as e:
            print(f"{i:>3}. ERROR: {e}  (is uvicorn running?)")
            continue

        mocked += r["mocked"]
        expected = EXPECTED_ROUTE.get(label)
        if expected:
            labeled += 1
            ok = r["route"] == expected
            correct += ok
            mark = "OK  " if ok else "MISS"
            if not ok:
                misses.append((i, label, r["route"], r["complexity_score"], prompt))
        else:
            mark = "    "

        flag = " [MOCK]" if r["mocked"] else ""
        short = prompt.replace("\n", " ")[:60]
        print(f"{i:>3}. {mark} {r['route']:<9} score={r['complexity_score']:.2f}{flag}  {short}", flush=True)

        out.write(
            f"## {i}. {mark.strip() or '-'} | label: {label or '?'} | routed: {r['route']} "
            f"({r['model_used']}) | score {r['complexity_score']:.2f}{flag}\n\n"
            f"**Prompt:**\n\n{prompt}\n\n**Response:**\n\n{r['response']}\n\n---\n\n"
        )
        out.flush()

        if i < len(rows):
            time.sleep(args.delay)

    print("\n--- Summary ---")
    if labeled:
        print(f"Routing matched your label: {correct}/{labeled} ({correct / labeled:.0%})")
    if mocked:
        print(f"{mocked} request(s) fell back to MOCK (rate limit or API error) - their costs are fake.")
    if misses:
        print("\nMisrouted prompts:")
        for i, label, route, score, prompt in misses:
            print(f"  #{i} labeled {label}, routed {route} (score {score:.2f}): {prompt.replace(chr(10), ' ')[:70]}")
    out.close()
    print(f"\nAll responses saved to {out_path}")
    print("Open the dashboard to see the cost totals.")


if __name__ == "__main__":
    main()