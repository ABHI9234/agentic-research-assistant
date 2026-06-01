import streamlit as st
import httpx
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from frontend.utils.api_client import BASE_URL

st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")
st.title("📊 Evaluation Dashboard")

st.info("Enter questions about your uploaded documents. The system runs them through each strategy and scores with RAGAS metrics.")

with st.form("eval_form"):
    questions_raw = st.text_area(
        "Evaluation Questions (one per line)",
        height=150,
        placeholder="What is a qubit?\nHow does superposition work?\nWhat are the key principles?",
    )
    strategies = st.multiselect(
        "Strategies to Compare",
        ["vector", "graph", "hybrid"],
        default=["vector", "hybrid"],
    )
    submitted = st.form_submit_button("▶️ Run Evaluation")

if submitted and questions_raw.strip() and strategies:
    questions = [q.strip() for q in questions_raw.strip().splitlines() if q.strip()]
    st.info(f"Running {len(questions)} question(s) across {strategies}. This takes 1-2 minutes...")

    with st.spinner("Running evaluation pipeline..."):
        try:
            r = httpx.post(
                f"{BASE_URL}/eval/run",
                json={
                    "questions": questions,
                    "strategies": strategies,
                },
                timeout=300.0,
            )
            result = r.json()

            if "detail" in result:
                st.error(f"Evaluation failed: {result['detail']}")
            else:
                st.success(f"✅ {result.get('summary', 'Evaluation complete')}")

                results = result.get("results", [])
                if results:
                    # Metrics chart
                    metrics = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
                    metric_labels = ["Faithfulness", "Context Precision", "Context Recall", "Answer Relevancy"]
                    colors = {"vector": "#3498db", "graph": "#e74c3c", "hybrid": "#2ecc71"}

                    fig = go.Figure()
                    for r_item in results:
                        strategy = r_item["strategy"]
                        values = [r_item.get(m, 0) for m in metrics]
                        fig.add_trace(go.Bar(
                            name=strategy.capitalize(),
                            x=metric_labels,
                            y=values,
                            marker_color=colors.get(strategy, "#95a5a6"),
                        ))

                    fig.update_layout(
                        barmode="group",
                        yaxis_range=[0, 1],
                        yaxis_title="Score",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", y=1.1),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Detailed table
                    st.subheader("Detailed Results")
                    import pandas as pd
                    df = pd.DataFrame([{
                        "Strategy": r_item["strategy"].capitalize(),
                        "Faithfulness": f"{r_item.get('faithfulness', 0):.3f}",
                        "Precision": f"{r_item.get('context_precision', 0):.3f}",
                        "Recall": f"{r_item.get('context_recall', 0):.3f}",
                        "Relevancy": f"{r_item.get('answer_relevancy', 0):.3f}",
                        "Hallucination": f"{r_item.get('hallucination_rate', 0):.3f}",
                        "Latency (ms)": f"{r_item.get('avg_latency_ms', 0):.0f}",
                    } for r_item in results])
                    st.dataframe(df, use_container_width=True)

                    best = result.get("best_strategy", "hybrid")
                    st.success(f"🏆 Best strategy: **{best.capitalize()}**")

        except httpx.ReadTimeout:
            st.error("Evaluation timed out. Try fewer questions or use vector-only strategy.")
        except Exception as e:
            st.error(f"Error: {e}")

# Sample chart when no evaluation has been run
else:
    st.subheader("Strategy Comparison (Sample Data)")
    fig = go.Figure()
    metrics = ["Faithfulness", "Context Precision", "Context Recall", "Answer Relevancy"]
    sample = {
        "Vector RAG": [0.72, 0.68, 0.61, 0.74],
        "Graph RAG": [0.69, 0.71, 0.75, 0.70],
        "Hybrid RAG": [0.81, 0.79, 0.82, 0.83],
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
        yaxis_title="Score",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Sample data shown. Run a live evaluation above to replace with real RAGAS scores.")
