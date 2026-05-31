from fastapi import APIRouter
from fastapi.responses import JSONResponse
from loguru import logger
import time

from backend.app.config import get_settings

router = APIRouter()
settings = get_settings()

# Track server start time for uptime calculation
_start_time = time.time()


@router.get("/health")
async def health_check():
    """Basic health check — confirms the API is running."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "mode": settings.app_mode,
        "uptime_seconds": round(time.time() - _start_time, 2),
    }


@router.get("/health/detailed")
async def detailed_health():
    """
    Checks connectivity to Qdrant, Neo4j, and Groq.
    Used by the System Monitoring dashboard in Streamlit.
    """
    from qdrant_client import QdrantClient
    from neo4j import GraphDatabase
    from groq import Groq

    results = {
        "api": "healthy",
        "qdrant": "unknown",
        "neo4j": "unknown",
        "groq": "unknown",
    }

    # Check Qdrant
    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=5,
        )
        client.get_collections()
        results["qdrant"] = "healthy"
    except Exception as e:
        results["qdrant"] = f"unhealthy: {str(e)[:80]}"
        logger.warning(f"Qdrant health check failed: {e}")

    # Check Neo4j
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        results["neo4j"] = "healthy"
    except Exception as e:
        results["neo4j"] = f"unhealthy: {str(e)[:80]}"
        logger.warning(f"Neo4j health check failed: {e}")

    # Check Groq
    try:
        client = Groq(api_key=settings.groq_api_key)
        client.models.list()
        results["groq"] = "healthy"
    except Exception as e:
        results["groq"] = f"unhealthy: {str(e)[:80]}"
        logger.warning(f"Groq health check failed: {e}")

    overall = "healthy" if all(
        v == "healthy" for v in results.values()
    ) else "degraded"

    return {
        "status": overall,
        "services": results,
        "uptime_seconds": round(time.time() - _start_time, 2),
    }


@router.get("/metrics")
async def metrics():
    """
    Basic system metrics endpoint.
    Used by the System Monitoring dashboard in Streamlit.
    """
    import psutil

    try:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        return {
            "uptime_seconds": round(time.time() - _start_time, 2),
            "cpu_percent": cpu,
            "memory_used_mb": round(mem.used / 1024 / 1024, 1),
            "memory_total_mb": round(mem.total / 1024 / 1024, 1),
            "memory_percent": mem.percent,
        }
    except Exception as e:
        return {"error": str(e)}
