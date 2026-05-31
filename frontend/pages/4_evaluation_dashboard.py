import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")
st.title("📊 Evaluation Dashboard")

st.info("Run a RAGAS evaluation by submitting questions below. Results will appear here.")

with st.form("eval_form"):
    questions_raw = st.text_area(
        "Evaluation Questions (one per line)",
        height=150,
        placeholder="What is the main topic of the document?\nWhat methods are described?",
    )
    strategies = st.multiselect(
        "Strategies to Compare",
        ["vector", "graph", "hybrid"],
        default=["vector", "hybrid"],
    )
    submitted = st.form_submit_button("▶️ Run Evaluation")

if submitted and questions_raw.strip():
    questions = [q.strip() for q in questions_raw.strip().splitlines() if q.strip()]
    st.info(f"Evaluation with {len(questions)} questions across {strategies} strategies. "
            f"Full RAGAS integration coming in Step 12.")

# ── Sample comparison chart (placeholder until RAGAS runs) ───────────────────
st.subheader("Strategy Comparison (Sample)")
fig = go.Figure()
metrics = ["Faithfulness", "Context Precision", "Context Recall", "Answer Relevancy"]
sample = {
    "Vector RAG":   [0.72, 0.68, 0.61, 0.74],
    "Graph RAG":    [0.69, 0.71, 0.75, 0.70],
    "Hybrid RAG":   [0.81, 0.79, 0.82, 0.83],
}
colors = {"Vector RAG": "#3498db", "Graph RAG": "#e74c3c", "Hybrid RAG": "#2ecc71"}

for strategy, values in sample.items():
    fig.add_trace(go.Bar(
        name=strategy, x=metrics, y=values,
        marker_color=colors[strategy],
    ))

fig.update_layout(
    barmode="group", yaxis_range=[0, 1],
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="white",
    yaxis_title="Score",
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Sample data shown. Run a live evaluation above to replace with real RAGAS scores.")
