import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "router_log.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt_preview TEXT NOT NULL,
    complexity_score REAL NOT NULL,
    route TEXT NOT NULL,
    model_used TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    actual_cost REAL NOT NULL,
    baseline_cost REAL NOT NULL,
    mocked INTEGER NOT NULL
);
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def log_query(prompt, complexity_score, route, model_used,
              input_tokens, output_tokens, actual_cost, baseline_cost, mocked):
    conn = get_conn()
    conn.execute(
        """INSERT INTO queries
           (timestamp, prompt_preview, complexity_score, route, model_used,
            input_tokens, output_tokens, actual_cost, baseline_cost, mocked)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            prompt[:120],
            complexity_score,
            route,
            model_used,
            input_tokens,
            output_tokens,
            actual_cost,
            baseline_cost,
            int(mocked),
        ),
    )
    conn.commit()
    conn.close()


def get_stats():
    conn = get_conn()
    cur = conn.execute(
        """SELECT COUNT(*), COALESCE(SUM(actual_cost),0), COALESCE(SUM(baseline_cost),0),
                  SUM(CASE WHEN route='cheap' THEN 1 ELSE 0 END),
                  SUM(CASE WHEN route='expensive' THEN 1 ELSE 0 END)
           FROM queries"""
    )
    total, actual, baseline, n_cheap, n_expensive = cur.fetchone()
    conn.close()
    savings = baseline - actual
    pct_savings = (savings / baseline * 100) if baseline > 0 else 0.0
    return {
        "total_queries": total or 0,
        "total_actual_cost": actual or 0.0,
        "total_baseline_cost": baseline or 0.0,
        "total_savings": savings or 0.0,
        "pct_savings": pct_savings,
        "n_cheap": n_cheap or 0,
        "n_expensive": n_expensive or 0,
    }


def get_recent(limit=25):
    conn = get_conn()
    cur = conn.execute(
        """SELECT timestamp, prompt_preview, complexity_score, route, model_used,
                  actual_cost, baseline_cost
           FROM queries ORDER BY id DESC LIMIT ?""",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows
