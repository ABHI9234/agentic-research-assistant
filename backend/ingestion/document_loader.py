from pypdf import PdfReader
from pathlib import Path
from typing import Generator
from loguru import logger
import hashlib


def generate_doc_id(filepath: str) -> str:
    """Stable doc ID based on filename — same file always gets same ID."""
    return hashlib.md5(Path(filepath).name.encode()).hexdigest()


def stream_pages(filepath: str) -> Generator[dict, None, None]:
    """
    Generator that yields one page at a time from a PDF.
    Never loads the whole PDF into memory.

    Yields:
        {
            "doc_id": str,
            "filename": str,
            "page_num": int,
            "total_pages": int,
            "text": str,
        }
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF files are supported. Got: {path.suffix}")

    doc_id = generate_doc_id(filepath)
    filename = path.name

    try:
        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
        logger.info(f"Streaming {filename} — {total_pages} pages")

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
                text = text.strip()
                if not text:
                    logger.debug(f"Page {page_num} of {filename} is empty — skipping")
                    continue
                yield {
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_num": page_num,
                    "total_pages": total_pages,
                    "text": text,
                }
            except Exception as e:
                logger.warning(f"Could not extract page {page_num} of {filename}: {e}")
                continue

    except Exception as e:
        logger.error(f"Failed to open {filename}: {e}")
        raise
