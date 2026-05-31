from typing import List
from loguru import logger

from backend.app.config import get_settings
from backend.retrieval.vector_retriever import vector_search
from backend.retrieval.graph_retriever import graph_search

settings = get_settings()


def hybrid_search(
    query: str,
    top_k: int = None,
    depth: int = None,
    doc_ids: List[str] = None,
    vector_weight: float = None,
    graph_weight: float = None,
) -> List[dict]:
    """
    Combines vector and graph retrieval results using weighted score fusion.

    Why hybrid:
    - Vector catches semantic similarity (what the text means)
    - Graph catches relational context (how entities connect)
    - Together they cover both similarity AND structure

    Returns top_k de-duplicated chunks ranked by fused score.
    """
    if top_k is None:
        top_k = settings.retrieval_top_k
    if depth is None:
        depth = settings.graph_depth
    if vector_weight is None:
        vector_weight = settings.hybrid_vector_weight
    if graph_weight is None:
        graph_weight = settings.hybrid_graph_weight

    # Run both retrievers
    vector_results = vector_search(query, top_k=top_k * 2, doc_ids=doc_ids)
    graph_results = graph_search(query, top_k=top_k * 2, depth=depth)

    # Build a unified dict keyed by chunk_id for deduplication
    fused: dict = {}

    for chunk in vector_results:
        cid = chunk["chunk_id"]
        if cid not in fused:
            fused[cid] = chunk.copy()
            fused[cid]["vector_score"] = chunk["relevance_score"]
            fused[cid]["graph_score"] = 0.0
        else:
            fused[cid]["vector_score"] = chunk["relevance_score"]

    for chunk in graph_results:
        cid = chunk["chunk_id"]
        if cid not in fused:
            fused[cid] = chunk.copy()
            fused[cid]["vector_score"] = 0.0
            fused[cid]["graph_score"] = chunk["relevance_score"]
        else:
            fused[cid]["graph_score"] = chunk["relevance_score"]

    # Calculate fused score and set retrieval method
    for cid, chunk in fused.items():
        v = chunk.get("vector_score", 0.0)
        g = chunk.get("graph_score", 0.0)
        chunk["relevance_score"] = round(
            vector_weight * v + graph_weight * g, 4
        )
        # Label how this chunk was found
        if v > 0 and g > 0:
            chunk["retrieval_method"] = "hybrid"
        elif v > 0:
            chunk["retrieval_method"] = "vector"
        else:
            chunk["retrieval_method"] = "graph"

    # Sort by fused score descending, return top_k
    ranked = sorted(fused.values(), key=lambda x: x["relevance_score"], reverse=True)
    final = ranked[:top_k]

    logger.debug(
        f"Hybrid search: '{query[:60]}' → "
        f"{len(vector_results)}V + {len(graph_results)}G → {len(final)} fused"
    )
    return final
