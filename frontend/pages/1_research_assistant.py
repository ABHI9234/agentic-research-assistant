import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from frontend.utils.api_client import run_research

st.set_page_config(page_title="Research Assistant", page_icon="🧠", layout="wide")
st.title("🧠 Research Assistant")

if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "run_query" not in st.session_state:
    st.session_state.run_query = False

with st.sidebar:
    st.header("Query Settings")
    top_k = st.slider("Evidence Chunks (top_k)", 3, 15, 5)
    graph_depth = st.slider("Graph Depth", 1, 4, 2)
    enable_reflection = st.toggle("Enable Reflection Agent", value=True)
    enable_critic = st.toggle("Enable Critic Agent", value=True)

    st.divider()
    st.header("Query History")
    if st.session_state.query_history:
        for i, h in enumerate(reversed(st.session_state.query_history[-10:])):
            if st.button(f"↩ {h['query'][:35]}", key=f"hist_{i}"):
                st.session_state.current_query = h["query"]
                st.session_state.run_query = True
                st.rerun()
    else:
        st.caption("No queries yet.")

query = st.text_input(
    "Enter your research question",
    value=st.session_state.current_query,
    placeholder="Type your question and press Enter",
    key="query_input",
)

if query:
    st.session_state.current_query = query

run_now = st.session_state.run_query or st.button("🔍 Run Research", type="primary", disabled=not query.strip())
st.session_state.run_query = False

if run_now and st.session_state.current_query.strip():
    with st.spinner("Running 6-agent research pipeline..."):
        result = run_research(
            query=st.session_state.current_query,
            strategy="hybrid",
            top_k=top_k,
            graph_depth=graph_depth,
            enable_reflection=enable_reflection,
            enable_critic=enable_critic,
        )

    st.session_state.query_history.append({
        "query": st.session_state.current_query,
        "answer": result.get("answer", ""),
        "latency_ms": result.get("latency_ms", 0),
    })

    if "error" in result and not result.get("answer"):
        st.error(f"Error: {result['error']}")
    else:
        st.subheader("Answer")
        st.markdown(result.get("answer", "No answer generated."))

        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", f"{result.get('confidence_score', 0):.0%}")
        col2.metric("Latency", f"{result.get('latency_ms', 0):.0f} ms")
        col3.metric("Evidence Chunks", len(result.get("evidence_chunks", [])))

        with st.expander("🤖 Agent Decisions"):
            for d in result.get("agent_decisions", []):
                st.markdown(f"**{d['agent']}** — {d['decision']}")
                st.caption(f"Reasoning: {d['reasoning']} | Confidence: {d['confidence']:.0%}")
                st.divider()

        with st.expander("📄 Evidence Chunks"):
            for i, chunk in enumerate(result.get("evidence_chunks", []), 1):
                st.markdown(
                    f"**[{i}] {chunk['source_file']} — p.{chunk['page_num']}** "
                    f"*(score: {chunk['relevance_score']:.3f})*"
                )
                st.text(chunk["text"][:400])
                st.divider()

        with st.expander("📎 Citations"):
            for c in result.get("citations", []):
                st.markdown(f"**[{c['citation_id']}]** {c['source_file']}, p.{c['page_num']}")

        with st.expander("🔄 Reflection & Critic Output"):
            st.markdown(f"**Reflection:** {result.get('reflection_output', 'N/A')}")
            st.markdown(f"**Critic:** {result.get('critic_findings', 'N/A')}")

        with st.expander("📋 Full Markdown Report"):
            st.markdown(result.get("report_markdown", ""))

        usage = result.get("token_usage", {})
        if usage:
            with st.expander("🔢 Token Usage"):
                st.json(usage)
