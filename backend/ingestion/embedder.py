from typing import List, Generator
from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
import uuid

from backend.app.config import get_settings

settings = get_settings()

# Load embedding model once at module level — not on every call
# all-MiniLM-L6-v2: 384 dimensions, fast, runs on CPU, no API key needed
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded.")
    return _embedding_model


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def ensure_collections_exist(client: QdrantClient) -> None:
    """Creates Qdrant collections if they don't already exist."""
    existing = [c.name for c in client.get_collections().collections]

    if settings.qdrant_collection_chunks not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection_chunks,
            vectors_config=VectorParams(
                size=settings.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created collection: {settings.qdrant_collection_chunks}")

    if settings.qdrant_collection_entities not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection_entities,
            vectors_config=VectorParams(
                size=settings.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created collection: {settings.qdrant_collection_entities}")


def embed_and_store_chunks(
    chunk_generator: Generator[dict, None, None],
    job_id: str,
) -> dict:
    """
    Consumes chunk generator in batches.
    Embeds each batch and immediately writes to Qdrant.
    Never holds more than batch_size chunks in memory.

    Returns:
        {"chunks_stored": int, "batches_processed": int}
    """
    model = get_embedding_model()
    client = get_qdrant_client()
    ensure_collections_exist(client)

    batch: List[dict] = []
    chunks_stored = 0
    batches_processed = 0

    def flush_batch(b: List[dict]) -> None:
        nonlocal chunks_stored, batches_processed
        texts = [c["text"] for c in b]
        vectors = model.encode(texts, show_progress_bar=False).tolist()

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "filename": chunk["filename"],
                    "page_num": chunk["page_num"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "chunk_type": chunk["chunk_type"],
                    "job_id": job_id,
                },
            )
            for chunk, vector in zip(b, vectors)
        ]

        client.upsert(
            collection_name=settings.qdrant_collection_chunks,
            points=points,
        )
        chunks_stored += len(b)
        batches_processed += 1
        logger.debug(f"Stored batch {batches_processed} — {len(b)} chunks ({chunks_stored} total)")

    for chunk in chunk_generator:
        batch.append(chunk)
        if len(batch) >= settings.batch_size:
            flush_batch(batch)
            batch = []

    # Flush any remaining chunks
    if batch:
        flush_batch(batch)

    logger.info(f"Embedding complete — {chunks_stored} chunks in {batches_processed} batches")
    return {"chunks_stored": chunks_stored, "batches_processed": batches_processed}
