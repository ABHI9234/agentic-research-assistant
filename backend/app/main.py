import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from backend.app.config import get_settings
from backend.app.routers import (
    ingest_router,
    query_router,
    graph_router,
    eval_router,
    health_router,
)

settings = get_settings()


# ── Set LangSmith env vars before any langchain import ───────────────────────
for key, value in settings.get_langchain_env().items():
    os.environ[key] = value


# ── Logging setup ────────────────────────────────────────────────────────────
logger.remove()
if settings.is_development:
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="DEBUG",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
        colorize=True,
    )
else:
    logger.add(
        sink="logs/app.log",
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} — {message}",
    )


# ── Lifespan: startup and shutdown events ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Mode: {settings.app_mode}")
    logger.info(f"Groq model: {settings.groq_model}")

    # Create upload directory if it doesn't exist
    os.makedirs(settings.upload_dir, exist_ok=True)
    logger.info(f"Upload directory ready: {settings.upload_dir}")

    # Validate critical env vars
    missing = []
    if not settings.groq_api_key:
        missing.append("GROQ_API_KEY")
    if not settings.qdrant_url:
        missing.append("QDRANT_URL")
    if not settings.neo4j_uri:
        missing.append("NEO4J_URI")

    if missing:
        logger.warning(f"Missing env vars: {missing}. Some features will not work.")
    else:
        logger.info("All critical env vars present.")

    yield

    # SHUTDOWN
    logger.info("Shutting down — cleaning up resources.")


# ── FastAPI app instance ──────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-grade Agentic GraphRAG Research Assistant",
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    lifespan=lifespan,
)


# ── CORS middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else [settings.backend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(duration)
    if settings.is_development:
        logger.debug(f"{request.method} {request.url.path} → {response.status_code} ({duration}ms)")
    return response


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.is_development else "Contact support.",
            "path": str(request.url.path),
        },
    )


# ── Register all routers ──────────────────────────────────────────────────────
app.include_router(health_router.router, prefix="/api", tags=["Health"])
app.include_router(ingest_router.router, prefix="/api/ingest", tags=["Ingestion"])
app.include_router(query_router.router, prefix="/api/query", tags=["Query"])
app.include_router(graph_router.router, prefix="/api/graph", tags=["Graph"])
app.include_router(eval_router.router, prefix="/api/eval", tags=["Evaluation"])


# ── Root endpoint ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "mode": settings.app_mode,
        "docs": "/docs",
        "health": "/api/health",
    }


# Preload embedding model on startup to avoid cold-start timeout on first upload
@app.on_event("startup")
async def preload_models():
    try:
        from backend.ingestion.embedder import get_embedding_model
        get_embedding_model()
        logger.info("Embedding model preloaded successfully.")
    except Exception as e:
        logger.warning(f"Could not preload embedding model: {e}")
