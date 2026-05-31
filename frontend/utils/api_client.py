import httpx
import os

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000") + "/api"
TIMEOUT = 120.0


def get_health():
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        return r.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


def get_detailed_health():
    try:
        r = httpx.get(f"{BASE_URL}/health/detailed", timeout=10)
        return r.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


def get_metrics():
    try:
        r = httpx.get(f"{BASE_URL}/metrics", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def upload_document(file_bytes: bytes, filename: str) -> dict:
    try:
        r = httpx.post(
            f"{BASE_URL}/ingest/upload",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=TIMEOUT,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_job_status(job_id: str) -> dict:
    try:
        r = httpx.get(f"{BASE_URL}/ingest/status/{job_id}", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def list_documents() -> dict:
    try:
        r = httpx.get(f"{BASE_URL}/ingest/documents", timeout=10)
        return r.json()
    except Exception as e:
        return {"documents": [], "total": 0}


def run_research(
    query: str,
    strategy: str = "hybrid",
    top_k: int = 5,
    graph_depth: int = 2,
    enable_reflection: bool = True,
    enable_critic: bool = True,
) -> dict:
    try:
        r = httpx.post(
            f"{BASE_URL}/query/research",
            json={
                "query": query,
                "retrieval_strategy": strategy,
                "top_k": top_k,
                "graph_depth": graph_depth,
                "enable_reflection": enable_reflection,
                "enable_critic": enable_critic,
            },
            timeout=TIMEOUT,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_graph(limit: int = 100) -> dict:
    try:
        r = httpx.get(f"{BASE_URL}/graph/explore?limit={limit}", timeout=30)
        return r.json()
    except Exception as e:
        return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0}


def traverse_graph(start_entity: str, depth: int = 2) -> dict:
    try:
        r = httpx.post(
            f"{BASE_URL}/graph/traverse",
            json={"start_entity": start_entity, "max_depth": depth},
            timeout=30,
        )
        return r.json()
    except Exception as e:
        return {"nodes": [], "edges": [], "paths": []}


def clear_all_data() -> dict:
    try:
        r = httpx.delete(f"{BASE_URL}/ingest/clear-all", timeout=30)
        return r.json()
    except Exception as e:
        return {"error": str(e)}
