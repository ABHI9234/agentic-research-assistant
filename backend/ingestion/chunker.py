from typing import Generator, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
import uuid

from backend.app.config import get_settings

settings = get_settings()


def get_splitter() -> RecursiveCharacterTextSplitter:
    """Creates a text splitter using settings from config."""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_page(page: dict) -> Generator[dict, None, None]:
    """
    Takes a single page dict from stream_pages() and yields chunks.
    Each chunk carries full metadata needed for Qdrant and Neo4j.

    Yields:
        {
            "chunk_id": str,
            "doc_id": str,
            "filename": str,
            "page_num": int,
            "total_pages": int,
            "chunk_index": int,
            "text": str,
            "chunk_type": "text",
        }
    """
    splitter = get_splitter()
    text = page["text"]

    if not text.strip():
        return

    chunks: List[str] = splitter.split_text(text)

    for chunk_index, chunk_text in enumerate(chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue
        yield {
            "chunk_id": str(uuid.uuid4()),
            "doc_id": page["doc_id"],
            "filename": page["filename"],
            "page_num": page["page_num"],
            "total_pages": page["total_pages"],
            "chunk_index": chunk_index,
            "text": chunk_text,
            "chunk_type": "text",
        }


def stream_chunks(filepath: str) -> Generator[dict, None, None]:
    """
    Full pipeline: PDF path → stream pages → chunk each page → yield chunks.
    This is the main entry point used by the ingestion service.
    Memory usage stays flat regardless of PDF size.
    """
    from backend.ingestion.document_loader import stream_pages

    chunk_count = 0
    for page in stream_pages(filepath):
        for chunk in chunk_page(page):
            chunk_count += 1
            yield chunk

    logger.info(f"Chunking complete: {chunk_count} chunks from {filepath}")
