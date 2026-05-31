from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class RetrievalStrategy(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = Field(default=5, ge=1, le=20)
    graph_depth: int = Field(default=2, ge=1, le=4)
    enable_reflection: bool = True
    enable_critic: bool = True
    doc_ids: Optional[List[str]] = None  # filter to specific documents


class EvidenceChunk(BaseModel):
    chunk_id: str
    text: str
    source_file: str
    page_num: int
    relevance_score: float
    retrieval_method: str  # "vector", "graph", or "hybrid"


class Citation(BaseModel):
    citation_id: str
    source_file: str
    page_num: int
    chunk_text: str
    claim: str


class AgentDecision(BaseModel):
    agent: str
    decision: str
    reasoning: str
    confidence: float


class ResearchResponse(BaseModel):
    query: str
    answer: str
    confidence_score: float
    retrieval_strategy_used: RetrievalStrategy
    evidence_chunks: List[EvidenceChunk]
    citations: List[Citation]
    agent_decisions: List[AgentDecision]
    reflection_output: Optional[str] = None
    critic_findings: Optional[str] = None
    report_markdown: Optional[str] = None
    latency_ms: float
    token_usage: Dict[str, int] = {}


class QueryHistoryItem(BaseModel):
    query_id: str
    query: str
    answer_preview: str
    timestamp: str
    latency_ms: float
    strategy: str
