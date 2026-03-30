"""QuantMind FastAPI application — Phase 3 entry point.

Starts the FastAPI server that exposes the LangGraph trading advisor
as a REST API. The MERN frontend (Phase 4) will call this API.

Endpoints:
    GET  /health          → Health check
    POST /analyze         → Full trading analysis (LangGraph workflow)

To run:
    cd quantmind/backend
    source .venv/bin/activate
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Then test:
    curl -X POST http://localhost:8000/analyze \
      -H "Content-Type: application/json" \
      -d '{"ticker": "AAPL", "query": "Should I buy Apple stock?"}'

API docs (auto-generated):
    http://localhost:8000/docs      ← Swagger UI
    http://localhost:8000/redoc     ← ReDoc
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.analysis import router as analysis_router
from api.schemas import HealthResponse
from config.logging_config import get_logger, setup_logging, stop_logging
from config.settings import settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager.

    Runs setup on startup and cleanup on shutdown.
    Using lifespan instead of deprecated @app.on_event decorators.
    """
    # --- Startup ---
    setup_logging(async_mode=False)  # Sync mode for simplicity in dev
    logger.info(
        "QuantMind API starting | host=%s | port=%d | debug=%s",
        settings.fastapi_host,
        settings.fastapi_port,
        settings.debug,
    )
    logger.info(
        "Groq model: %s | Groq key configured: %s",
        settings.groq_model,
        bool(settings.groq_api_key),
    )

    yield  # Application runs here

    # --- Shutdown ---
    logger.info("QuantMind API shutting down")
    stop_logging()


# Create FastAPI application
app = FastAPI(
    title="QuantMind API",
    description="""
## QuantMind — AI-Powered Algorithmic Trading Strategy Advisor

Phase 3: LangGraph Multi-Agent Orchestration

### What it does
Analyzes stocks using a 5-agent LangGraph pipeline:
1. **ResearchAgent** — Fetches real-time market data (yfinance, free)
2. **RAGAgent** — Retrieves SEC filings & news (ChromaDB + sentence-transformers)
3. **StrategyAgent** — Selects optimal strategy (Momentum or MeanReversion)
4. **BacktestAgent** — Runs historical backtest with DSA-optimized engine
5. **RiskAgent** — Evaluates Sharpe, VaR, Drawdown limits
6. **ExplainerAgent** — Generates cited explanation (Groq Llama 3.1 70B)

### Free to use
- No API keys required for basic functionality
- Add GROQ_API_KEY for AI-powered explanations (free at console.groq.com)
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# --- CORS middleware (allows React frontend to call this API) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React dev server
        "http://localhost:5173",   # Vite dev server
        "https://quantmind.vercel.app",  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Register routes ---
app.include_router(analysis_router)


# --- Health check endpoint ---
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
)
async def health() -> HealthResponse:
    """Check if the API is running.

    Returns:
        HealthResponse with status, version, and phase.
    """
    return HealthResponse(
        status="ok",
        version="1.0.0",
        phase="Phase 3 — LangGraph Agents",
    )


# --- Root redirect to docs ---
@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """Redirect root to API docs."""
    return JSONResponse(
        content={
            "message": "QuantMind API v1.0.0",
            "docs": "/docs",
            "health": "/health",
            "analyze": "POST /analyze",
        }
    )


# --- Run directly (for development) ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=settings.debug,
        log_level="info",
    )
