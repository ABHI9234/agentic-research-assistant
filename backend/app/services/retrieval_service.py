import time
from loguru import logger
from backend.graph.workflow import compiled_workflow
from backend.graph.state import ResearchState
from backend.app.schemas.query_schemas import (
    ResearchRequest,
    ResearchResponse,
    EvidenceChunk,
    Citation,
    AgentDecision,
    RetrievalStrategy,
)


async def run_research_query(request: ResearchRequest) -> ResearchResponse:
    """
    Entry point called by the query router.
    Builds initial state, runs the LangGraph workflow, returns structured response.
    """
    start_time = time.time()

    # Build initial state
    initial_state: ResearchState = {
        "query": request.query,
        "retrieval_strategy": request.retrieval_strategy.value,
        "top_k": request.top_k,
        "graph_depth": request.graph_depth,
        "enable_reflection": request.enable_reflection,
        "enable_critic": request.enable_critic,
        "doc_ids": request.doc_ids,
        # Outputs — start empty
        "subqueries": [],
        "retrieval_plan": {},
        "evidence_chunks": [],
        "retrieval_method_used": "",
        "reflection_passed": False,
        "reflection_output": "",
        "reflection_iterations": 0,
        "critic_passed": False,
        "critic_findings": "",
        "citations": [],
        "answer": "",
        "report_markdown": "",
        "confidence_score": 0.0,
        "agent_decisions": [],
        "token_usage": {},
        "error": None,
    }

    logger.info(f"[Query] Starting workflow for: {request.query[:80]}")

    try:
        final_state = compiled_workflow.invoke(initial_state)
    except Exception as e:
        logger.error(f"[Query] Workflow failed: {e}")
        final_state = {**initial_state, "answer": f"Workflow error: {e}", "error": str(e)}

    latency_ms = round((time.time() - start_time) * 1000, 2)

    # Map evidence chunks to schema
    evidence_chunks = [
        EvidenceChunk(
            chunk_id=c.get("chunk_id", ""),
            text=c.get("text", ""),
            source_file=c.get("source_file", ""),
            page_num=c.get("page_num", 0),
            relevance_score=c.get("relevance_score", 0.0),
            retrieval_method=c.get("retrieval_method", ""),
        )
        for c in final_state.get("evidence_chunks", [])
    ]

    # Map citations to schema
    citations = [
        Citation(
            citation_id=c.get("citation_id", ""),
            source_file=c.get("source_file", ""),
            page_num=c.get("page_num", 0),
            chunk_text=c.get("chunk_text", ""),
            claim=c.get("claim", ""),
        )
        for c in final_state.get("citations", [])
    ]

    # Map agent decisions to schema
    agent_decisions = [
        AgentDecision(
            agent=d.get("agent", ""),
            decision=d.get("decision", ""),
            reasoning=d.get("reasoning", ""),
            confidence=d.get("confidence", 0.0),
        )
        for d in final_state.get("agent_decisions", [])
    ]

    strategy_used = RetrievalStrategy(
        final_state.get("retrieval_method_used", request.retrieval_strategy.value)
    )

    logger.info(f"[Query] Completed in {latency_ms}ms")

    return ResearchResponse(
        query=request.query,
        answer=final_state.get("answer", ""),
        confidence_score=final_state.get("confidence_score", 0.0),
        retrieval_strategy_used=strategy_used,
        evidence_chunks=evidence_chunks,
        citations=citations,
        agent_decisions=agent_decisions,
        reflection_output=final_state.get("reflection_output"),
        critic_findings=final_state.get("critic_findings"),
        report_markdown=final_state.get("report_markdown"),
        latency_ms=latency_ms,
        token_usage=final_state.get("token_usage", {}),
    )
