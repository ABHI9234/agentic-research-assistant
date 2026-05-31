import streamlit as st

st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 Agentic Enterprise Research Assistant")
st.markdown("""
Welcome to your production-grade GraphRAG research platform.

**Use the sidebar to navigate:**

| Page | Purpose |
|------|---------|
| 🧠 Research Assistant | Ask research questions across your documents |
| 📁 Document Management | Upload and track document ingestion |
| 🕸️ Knowledge Graph | Explore extracted entities and relationships |
| 📊 Evaluation Dashboard | Compare retrieval strategies with RAGAS |
| 🖥️ System Monitoring | Live API and database health |
| ⚙️ Settings | Configure model and retrieval parameters |
""")

st.info("Make sure the FastAPI backend is running on http://localhost:8000 before using any features.")
