from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.app.schemas.query_schemas import ResearchRequest, ResearchResponse
from backend.app.services.retrieval_service import run_research_query

router = APIRouter()


@router.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """
    Main research endpoint. Runs the full 6-agent LangGraph workflow.
    Accepts a query, returns answer + citations + agent trace.
    """
    try:
        result = await run_research_query(request)
        return result
    except Exception as e:
        logger.error(f"Research endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
