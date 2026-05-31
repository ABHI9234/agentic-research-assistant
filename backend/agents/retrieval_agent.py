from loguru import logger
from typing import List

from backend.app.config import get_settings
from backend.graph.state import ResearchState
from backend.retrieval.vector_retriever import vector_search
from backend.retrieval.graph_retriever import graph_search
from backend.retrieval.hybrid_retriever import hybrid_search

settings = get_settings()


def run_retrieval(state: ResearchState) -> ResearchState:
    """
    Retrieval Agent — ReAct style.
    Executes the retrieval plan from the Planner.
    Runs retrieval for each subquery and merges results.

    Why ReAct: The retrieval agent must decide at runtime which combination
    of vector/graph/hybrid to use per subquery, then rank and deduplicate
    evidence across all subqueries before passing it forward.
    """
    subqueries = state.get("subqueries") or [state["query"]]
    strategy = state.get("retrieval_strategy", "hybrid")
    top_k = state.get("top_k", settings.retrieval_top_k)
    depth = state.get("graph_depth", settings.graph_depth)
    doc_ids = state.get("doc_ids")

    logger.info(f"[Retrieval] Strategy: {strategy}, Subqueries: {len(subqueries)}")

    all_chunks: List[dict] = []
    seen_ids = set()

    for subquery in subqueries:
        if strategy == "vector":
            chunks = vector_search(subquery, top_k=top_k, doc_ids=doc_ids)
        elif strategy == "graph":
            chunks = graph_search(subquery, top_k=top_k, depth=depth)
        else:
            chunks = hybrid_search(
                subquery, top_k=top_k, depth=depth, doc_ids=doc_ids
            )

        for chunk in chunks:
            cid = chunk.get("chunk_id", "")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(chunk)

    # Sort all gathered evidence by relevance score
    all_chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    final_chunks = all_chunks[:top_k * 2]

    logger.info(f"[Retrieval] Total evidence chunks gathered: {len(final_chunks)}")

    state["evidence_chunks"] = final_chunks
    state["retrieval_method_used"] = strategy
    state["agent_decisions"].append(
        {
            "agent": "Retrieval",
            "decision": f"Gathered {len(final_chunks)} chunks via {strategy}",
            "reasoning": f"Ran {len(subqueries)} subqueries, deduplicated results",
            "confidence": 0.85,
        }
    )
    return state
