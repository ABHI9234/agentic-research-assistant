from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import uuid


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    message: str = "File received. Processing started."


class JobStatusResponse(BaseModel):
    job_id: str
    filename: str
    status: ProcessingStatus
    progress_percent: float = 0.0
    pages_processed: int = 0
    chunks_created: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class DocumentRecord(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunk_count: int
    entity_count: int
    upload_date: str
    status: ProcessingStatus
    file_size_kb: float


class DocumentListResponse(BaseModel):
    documents: List[DocumentRecord]
    total: int


class DeleteResponse(BaseModel):
    doc_id: str
    message: str
    success: bool
