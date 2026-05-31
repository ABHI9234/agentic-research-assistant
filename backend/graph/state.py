from typing import TypedDict, List, Optional, Dict, Any
from backend.app.schemas.query_schemas import (
    EvidenceChunk,
    Citation,
    AgentDecision,
    RetrievalStrategy,
)


class ResearchState(TypedDict):
    """
    The single shared state object passed between every node in the LangGraph.
    Each agent reads from it and writes its outputs back into it.
    """
    # Input
    query: str
    retrieval_strategy: str
    top_k: int
    graph_depth: int
    enable_reflection: bool
    enable_critic: bool
    doc_ids: Optional[List[str]]

    # Planner outputs
    subqueries: List[str]
    retrieval_plan: Dict[str, Any]

    # Retrieval outputs
    evidence_chunks: List[dict]
    retrieval_method_used: str

    # Reflection outputs
    reflection_passed: bool
    reflection_output: str
    reflection_iterations: int

    # Critic outputs
    critic_passed: bool
    critic_findings: str

    # Citation outputs
    citations: List[dict]

    # Report outputs
    answer: str
    report_markdown: str
    confidence_score: float

    # Agent trace (for UI display)
    agent_decisions: List[dict]

    # Meta
    token_usage: Dict[str, int]
    error: Optional[str]
