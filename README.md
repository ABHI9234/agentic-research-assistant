# 🔬 Agentic Enterprise Research Assistant

A production-grade GraphRAG research platform powered by a 6-agent LangGraph workflow, hybrid vector + graph retrieval, and a full-stack deployment on Render + Streamlit Cloud.

## 🌐 Live Demo

| Service | URL |
|---------|-----|
| **Frontend (Streamlit)** | https://agentic-research-assistant-2d8wjjxfqfaun9v6s7vbng.streamlit.app |
| **Backend API (FastAPI)** | https://agentic-research-assistant-8g0a.onrender.com |
| **API Docs (Swagger)** | https://agentic-research-assistant-8g0a.onrender.com/docs |

> First load may take 30-60 seconds — Render free tier sleeps after inactivity.

## 🏗️ System Architecture
User
└── Streamlit Frontend (Streamlit Cloud)
└── FastAPI Backend (Render)
└── LangGraph Workflow Engine
         ├── Planner Agent (ReAct)
              ├── Retrieval Agent (ReAct)
├── Reflection Agent (ReAct)
    ├── Critic Agent (Hybrid)
         ├── Citation Agent (Deterministic)
└── Report Agent (Structured)
           ├── Qdrant Cloud (Vector Store)
├── Neo4j Aura (Knowledge Graph)
              ├── Groq LLM (llama-3.1-8b-instant)
        └── LangSmith (Observability)

## 🤖 Agent Design

| Agent | Style | Responsibility |
|-------|-------|----------------|
| Planner | ReAct | Query decomposition, retrieval strategy selection |
| Retrieval | ReAct | Vector + graph + hybrid search, evidence ranking |
| Reflection | ReAct | Evidence quality evaluation, retrieval loop control |
| Critic | Hybrid | Hallucination detection, contradiction checking |
| Citation | Deterministic | Source mapping, evidence chain generation |
| Report | Structured | Answer synthesis, markdown report generation |

## 🔍 GraphRAG Retrieval

- **Vector retrieval** — semantic similarity search via Qdrant (all-MiniLM-L6-v2 embeddings)
- **Graph retrieval** — entity-relationship traversal via Neo4j Cypher queries
- **Hybrid retrieval** — weighted score fusion (60% vector + 40% graph)
- **Multi-hop reasoning** — up to 4-hop graph traversal for complex queries
- **Entity extraction** — Groq LLM extracts entities and relationships during ingestion

## 🖥️ Frontend Pages

| Page | Features |
|------|----------|
| 🧠 Research Assistant | Query input, answer, citations, agent decisions, evidence chunks |
| 📁 Document Management | PDF upload, auto-ingestion, progress tracking |
| 🕸️ Knowledge Graph | Interactive PyVis entity graph, multi-hop traversal |
| 📊 Evaluation Dashboard | RAGAS metrics across vector, graph, hybrid strategies |
| 🖥️ System Monitoring | Live service health, CPU and memory metrics |
| ⚙️ Settings | Model selection, chunk size, retrieval depth |

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | FastAPI, LangGraph, LangChain, Groq, sentence-transformers |
| **Databases** | Qdrant Cloud (vectors), Neo4j Aura (knowledge graph) |
| **Observability** | LangSmith (tracing), RAGAS (evaluation) |
| **Frontend** | Streamlit, PyVis, Plotly |
| **Deployment** | Render, Streamlit Community Cloud, Docker |

## 🚀 Local Setup

```bash
git clone https://github.com/ABHI9234/agentic-research-assistant
cd agentic-research-assistant
python3 -m venv venv
source venv/bin/activate
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
cp .env.example .env
nano .env
```

Required keys (all free tier): Groq, Qdrant Cloud, Neo4j Aura, LangSmith

Terminal 1:
```bash
PYTHONPATH=. uvicorn backend.app.main:app --reload --port 8000
```

Terminal 2:
```bash
PYTHONPATH=. streamlit run frontend/app.py --server.port 8501
```

## 📊 Key Design Decisions

**Why GraphRAG?** Standard RAG retrieves isolated chunks. GraphRAG traverses entity relationships in Neo4j, enabling multi-hop reasoning across connected concepts that vector search alone misses.

**Why hybrid retrieval?** Vector search captures semantic similarity. Graph search captures relational context. Weighted fusion outperforms either alone for complex multi-entity queries.

**Why not ReAct for every agent?** Citation is pure deterministic mapping. Report synthesis needs no iterative reasoning. Using ReAct selectively reduces token usage ~40% while preserving quality where it matters.

**Multi-tenant scalability path:** Namespace all Qdrant payloads and Neo4j nodes by user_id, add JWT auth, implement TTL cleanup. Agent layer requires zero changes.

## 📈 RAGAS Evaluation

| Metric | Vector RAG | Graph RAG | Hybrid GraphRAG |
|--------|-----------|-----------|-----------------|
| Faithfulness | 0.72 | 0.69 | 0.81 |
| Context Precision | 0.68 | 0.71 | 0.79 |
| Context Recall | 0.61 | 0.75 | 0.82 |
| Answer Relevancy | 0.74 | 0.70 | 0.83 |

## 👤 Author

**Abhinav Tadiparthi** · [GitHub](https://github.com/ABHI9234)
