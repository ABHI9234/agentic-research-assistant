import json
import os
from typing import Dict, Optional
from loguru import logger
from datetime import datetime
from backend.app.schemas.ingest_schemas import ProcessingStatus

# File-based persistence — survives server restarts within same deploy
STATE_FILE = "/tmp/job_state.json"

_jobs: Dict[str, dict] = {}


def _load_state():
    global _jobs
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                _jobs = json.load(f)
    except Exception:
        _jobs = {}


def _save_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(_jobs, f)
    except Exception as e:
        logger.warning(f"Could not save job state: {e}")


# Load on module import
_load_state()


def create_job(job_id: str, filename: str) -> dict:
    job = {
        "job_id": job_id,
        "filename": filename,
        "status": ProcessingStatus.PENDING,
        "progress_percent": 0.0,
        "pages_processed": 0,
        "chunks_created": 0,
        "entities_extracted": 0,
        "relationships_extracted": 0,
        "error_message": None,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }
    _jobs[job_id] = job
    _save_state()
    return job


def get_job(job_id: str) -> Optional[dict]:
    _load_state()
    return _jobs.get(job_id)


def get_all_jobs() -> Dict[str, dict]:
    _load_state()
    return dict(_jobs)


def update_job(job_id: str, **kwargs) -> None:
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)
        _save_state()


def mark_completed(job_id: str, chunks: int, entities: int, relationships: int) -> None:
    update_job(
        job_id,
        status=ProcessingStatus.COMPLETED,
        progress_percent=100.0,
        chunks_created=chunks,
        entities_extracted=entities,
        relationships_extracted=relationships,
        completed_at=datetime.utcnow().isoformat(),
    )
    logger.info(f"Job completed: {job_id} — {chunks} chunks, {entities} entities")


def mark_failed(job_id: str, error: str) -> None:
    update_job(
        job_id,
        status=ProcessingStatus.FAILED,
        error_message=error,
        completed_at=datetime.utcnow().isoformat(),
    )


async def run_ingestion_job(job_id: str, filepath: str) -> None:
    from backend.ingestion.chunker import stream_chunks
    from backend.ingestion.embedder import embed_and_store_chunks
    from backend.retrieval.entity_extractor import extract_and_store_entities

    update_job(job_id, status=ProcessingStatus.PROCESSING, progress_percent=10.0)

    try:
        chunk_gen = stream_chunks(filepath)
        embed_result = embed_and_store_chunks(chunk_gen, job_id)
        chunks_stored = embed_result["chunks_stored"]
        update_job(job_id, chunks_created=chunks_stored, progress_percent=60.0)

        entity_result = await extract_and_store_entities(filepath, job_id)
        entities = entity_result.get("entities_stored", 0)
        relationships = entity_result.get("relationships_stored", 0)

        mark_completed(job_id, chunks_stored, entities, relationships)

    except Exception as e:
        logger.exception(f"[{job_id}] Ingestion failed: {e}")
        mark_failed(job_id, str(e))
