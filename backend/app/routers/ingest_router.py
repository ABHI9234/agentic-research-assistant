import os
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from loguru import logger

from backend.app.config import get_settings
from backend.app.schemas.ingest_schemas import (
    UploadResponse, JobStatusResponse, DocumentListResponse,
    DocumentRecord, DeleteResponse, ProcessingStatus,
)
from backend.ingestion.queue import create_job, get_job, get_all_jobs, run_ingestion_job
import uuid

router = APIRouter()
settings = get_settings()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max {settings.max_file_size_mb} MB.")
    os.makedirs(settings.upload_dir, exist_ok=True)
    filepath = os.path.join(settings.upload_dir, file.filename)
    with open(filepath, "wb") as f:
        f.write(contents)
    job_id = str(uuid.uuid4())
    create_job(job_id, file.filename)
    background_tasks.add_task(run_ingestion_job, job_id, filepath)
    return UploadResponse(job_id=job_id, filename=file.filename,
                         status=ProcessingStatus.PENDING,
                         message="File received. Processing started.")


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return JobStatusResponse(**job)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    jobs = get_all_jobs()
    docs = []
    for job_id, job in jobs.items():
        filepath = os.path.join(settings.upload_dir, job["filename"])
        file_size_kb = round(os.path.getsize(filepath) / 1024, 1) if os.path.exists(filepath) else 0.0
        docs.append(DocumentRecord(
            doc_id=job_id, filename=job["filename"],
            pages=job.get("pages_processed", 0),
            chunk_count=job.get("chunks_created", 0),
            entity_count=job.get("entities_extracted", 0),
            upload_date=job.get("started_at", ""),
            status=job["status"], file_size_kb=file_size_kb,
        ))
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete("/document/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    jobs = get_all_jobs()
    if doc_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
    job = jobs[doc_id]
    filepath = os.path.join(settings.upload_dir, job["filename"])
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        logger.warning(f"Could not delete file: {e}")
    return DeleteResponse(doc_id=doc_id, message=f"Document {job['filename']} removed.", success=True)


@router.delete("/clear-all")
async def clear_all_data():
    """
    Wipes all data from Qdrant collections and Neo4j graph.
    Called when user wants a fresh session.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    from neo4j import GraphDatabase
    from backend.ingestion.queue import _jobs

    errors = []

    # Clear Qdrant — delete and recreate both collections
    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        for collection in [settings.qdrant_collection_chunks, settings.qdrant_collection_entities]:
            try:
                client.delete_collection(collection)
            except Exception:
                pass
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=settings.embedding_dimension, distance=Distance.COSINE),
            )
        logger.info("Qdrant collections cleared and recreated.")
    except Exception as e:
        errors.append(f"Qdrant: {str(e)}")

    # Clear Neo4j — delete all nodes and relationships
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()
        logger.info("Neo4j graph cleared.")
    except Exception as e:
        errors.append(f"Neo4j: {str(e)}")

    # Clear in-memory job store
    _jobs.clear()

    # Delete uploaded files
    try:
        if os.path.exists(settings.upload_dir):
            for f in os.listdir(settings.upload_dir):
                os.remove(os.path.join(settings.upload_dir, f))
    except Exception as e:
        errors.append(f"Files: {str(e)}")

    if errors:
        return {"status": "partial", "errors": errors}
    return {"status": "cleared", "message": "All data wiped. Ready for fresh session."}
