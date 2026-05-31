from typing import List
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from backend.app.config import get_settings
from backend.ingestion.embedder import get_embedding_model, get_qdrant_client

settings = get_settings()


def vector_search(
    query: str,
    top_k: int = None,
    doc_ids: List[str] = None,
) -> List[dict]:
    """
    Embeds the query and searches Qdrant for the most similar chunks.

    Args:
        query: The search query string
        top_k: Number of results to return (defaults to settings value)
        doc_ids: Optional list of doc_ids to filter results to

    Returns:
        List of chunk dicts with relevance scores
    """
    if top_k is None:
        top_k = settings.retrieval_top_k

    model = get_embedding_model()
    client = get_qdrant_client()

    query_vector = model.encode(query, show_progress_bar=False).tolist()

    # Build optional filter for specific documents
    search_filter = None
    if doc_ids:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
                for doc_id in doc_ids
            ]
        )

    try:
        results = client.search(
            collection_name=settings.qdrant_collection_chunks,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        )

        chunks = []
        for hit in results:
            payload = hit.payload or {}
            chunks.append(
                {
                    "chunk_id": payload.get("chunk_id", ""),
                    "text": payload.get("text", ""),
                    "source_file": payload.get("filename", ""),
                    "page_num": payload.get("page_num", 0),
                    "doc_id": payload.get("doc_id", ""),
                    "relevance_score": round(float(hit.score), 4),
                    "retrieval_method": "vector",
                }
            )

        logger.debug(f"Vector search: '{query[:60]}' → {len(chunks)} results")
        return chunks

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []
