import streamlit as st
import pandas as pd
import db

st.set_page_config(page_title="LLM Cost Router Dashboard", layout="wide")
st.title("LLM Cost Router — Dashboard")
st.caption("Live view of routing decisions and cost savings vs. an 'always use the expensive model' baseline.")

stats = db.get_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total queries", stats["total_queries"])
col2.metric("Actual cost", f"${stats['total_actual_cost']:.4f}")
col3.metric("Baseline cost (always expensive)", f"${stats['total_baseline_cost']:.4f}")
col4.metric("Savings", f"${stats['total_savings']:.4f}", f"{stats['pct_savings']:.1f}%")

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Route distribution")
    route_df = pd.DataFrame(
        {"route": ["cheap", "expensive"], "count": [stats["n_cheap"], stats["n_expensive"]]}
    )
    st.bar_chart(route_df.set_index("route"))

with right:
    st.subheader("Recent queries")
    rows = db.get_recent(25)
    if rows:
        df = pd.DataFrame(
            rows,
            columns=[
                "timestamp", "prompt", "complexity_score", "route",
                "model", "actual_cost", "baseline_cost",
            ],
        )
        st.dataframe(df, use_container_width=True, height=400)
    else:
        st.info("No queries logged yet. Send a POST /query request to the API first.")

st.divider()
st.caption(
    "Run `python train_classifier.py` once, then `uvicorn main:app --reload` to start sending queries."
)
