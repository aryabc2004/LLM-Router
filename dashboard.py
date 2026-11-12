import streamlit as st
import pandas as pd
import db

st.set_page_config(page_title="LLM Cost Router Dashboard", layout="wide")
st.title("LLM Cost Router — Dashboard")
st.caption("Routing decisions and cost savings vs. an 'always use the expensive model' baseline.")

# Mock-mode rows have fake token counts, so their costs are fake too.
real_only = st.toggle("Real API calls only (exclude mock rows)", value=True)

stats = db.get_stats(real_only=real_only)
if stats["n_mocked_in_db"] and real_only:
    st.caption(f"Hiding {stats['n_mocked_in_db']} mock-mode rows.")
elif stats["n_mocked_in_db"]:
    st.warning(f"{stats['n_mocked_in_db']} rows are from MOCK mode — these costs are not real.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total queries", stats["total_queries"])
col2.metric("Actual cost", f"${stats['total_actual_cost']:.4f}")
col3.metric("Baseline cost (always expensive)", f"${stats['total_baseline_cost']:.4f}")
col4.metric("Savings", f"${stats['total_savings']:.4f}", f"{stats['pct_savings']:.1f}%")

st.divider()

# --- Cumulative cost over time ---
st.subheader("Cumulative cost: actual vs. baseline")
series = db.get_cost_series(real_only=real_only)
if series:
    cost_df = pd.DataFrame(series, columns=["timestamp", "actual_cost", "baseline_cost"])
    cost_df["timestamp"] = pd.to_datetime(cost_df["timestamp"])
    cost_df = cost_df.set_index("timestamp")
    cumulative = pd.DataFrame({
        "Actual (routed)": cost_df["actual_cost"].cumsum(),
        "Baseline (always expensive)": cost_df["baseline_cost"].cumsum(),
    })
    st.line_chart(cumulative, y_label="Cumulative cost ($)")
else:
    st.info("No queries logged yet. Send a POST /query request to the API first.")

st.divider()

left, right = st.columns([1, 2])

with left:
    st.subheader("Route distribution")
    total = stats["n_cheap"] + stats["n_expensive"]
    if total:
        route_df = pd.DataFrame(
            {"queries": [stats["n_cheap"], stats["n_expensive"]]},
            index=["cheap", "expensive"],
        )
        st.bar_chart(route_df, horizontal=True, x_label="Queries")
        st.write(
            f"**{stats['n_cheap'] / total:.0%}** of queries went to the cheap model "
            f"({stats['n_cheap']} cheap / {stats['n_expensive']} expensive)."
        )
    else:
        st.info("No routed queries yet.")

with right:
    st.subheader("Recent queries")
    rows = db.get_recent(25, real_only=real_only)
    if rows:
        df = pd.DataFrame(
            rows,
            columns=[
                "timestamp", "prompt", "complexity_score", "route",
                "model", "actual_cost", "baseline_cost", "mocked",
            ],
        )
        df["mocked"] = df["mocked"].astype(bool)
        st.dataframe(df, width="stretch", height=400)
    else:
        st.info("No queries logged yet.")

st.divider()
st.caption(
    "Run `python train_classifier.py` once, then `uvicorn main:app --reload` to start sending queries."
)
