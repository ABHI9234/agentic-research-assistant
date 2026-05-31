from loguru import logger
import uuid

from backend.graph.state import ResearchState


def run_citation(state: ResearchState) -> ResearchState:
    """
    Citation Agent — Fully deterministic pipeline.

    Why NOT ReAct: Citation is pure source mapping.
    We already have the evidence chunks with metadata.
    No LLM reasoning needed — just build structured citations from what we have.
    """
    chunks = state.get("evidence_chunks", [])
    citations = []

    seen_sources = set()

    for chunk in chunks:
        source_file = chunk.get("source_file", "Unknown")
        page_num = chunk.get("page_num", 0)
        text = chunk.get("text", "")

        # Deduplicate by source+page
        source_key = f"{source_file}:{page_num}"
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)

        # Build citation
        citation = {
            "citation_id": str(uuid.uuid4())[:8],
            "source_file": source_file,
            "page_num": page_num,
            "chunk_text": text[:300],
            "claim": f"Information retrieved from {source_file}, page {page_num}.",
        }
        citations.append(citation)

    logger.info(f"[Citation] Generated {len(citations)} citations")

    state["citations"] = citations
    state["agent_decisions"].append(
        {
            "agent": "Citation",
            "decision": f"Generated {len(citations)} citations",
            "reasoning": "Deterministic source mapping from evidence chunks.",
            "confidence": 1.0,
        }
    )
    return state
