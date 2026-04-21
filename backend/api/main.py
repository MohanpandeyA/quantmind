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
from api.routes.portfolio import router as portfolio_router
from api.routes.compare import router as compare_router
from api.routes.optimize import router as optimize_router
from api.routes.alerts import router as alerts_router
from api.routes.earnings import router as earnings_router
from api.routes.ticker import router as ticker_router
from api.routes.walk_forward import router as walk_forward_router
from api.routes.live_chart import router as live_chart_router
from api.schemas import HealthResponse
from config.logging_config import get_logger, setup_logging, stop_logging
from config.settings import settings
from engine.ssl_fix import apply_ssl_fix

# Apply SSL fix immediately at import time (before any yfinance usage)
apply_ssl_fix()

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager.

    Runs setup on startup and cleanup on shutdown.
    Using lifespan instead of deprecated @app.on_event decorators.

    CRITICAL FIX 3 — PRE-LOAD EMBEDDING MODEL:
        Problem: sentence-transformers takes 6-8 seconds to load on first use.
        Without pre-loading, the FIRST user request after server restart
        waits 6-8 extra seconds while the model loads.

        Fix: Load the model during FastAPI startup (before any requests arrive).
        Subsequent requests find the model already loaded — instant response.

        This is the 'eager initialization' pattern vs 'lazy initialization'.
        For ML models that are always needed, eager is better.
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

    # Pre-load embedding model at startup (Critical Fix 3)
    # Runs in background thread so it doesn't block the startup
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        logger.info("Pre-loading sentence-transformers embedding model...")
        from rag.embeddings import EmbeddingModel
        model = EmbeddingModel()
        # Load model in thread pool (CPU-bound, ~6-8s)
        await loop.run_in_executor(None, model._load_model)
        logger.info(
            "Embedding model pre-loaded | model=%s | dims=%d",
            model.model_name, model.dimensions,
        )
    except Exception as e:
        logger.warning("Embedding model pre-load failed (will load on first request): %s", e)

    # Pre-load FinBERT sentiment model at startup
    # First load downloads ~440MB from HuggingFace and takes ~30-40s.
    # Pre-loading ensures the first analysis request is not slow.
    try:
        logger.info("Pre-loading FinBERT sentiment model...")
        from agents.sentiment_agent import _load_finbert
        await loop.run_in_executor(None, _load_finbert)
        logger.info("FinBERT sentiment model pre-loaded | model=ProsusAI/finbert")
    except Exception as e:
        logger.warning("FinBERT pre-load failed (will load on first sentiment analysis): %s", e)

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
app.include_router(portfolio_router)
app.include_router(compare_router)
app.include_router(optimize_router)
app.include_router(alerts_router)
app.include_router(earnings_router)
app.include_router(ticker_router)
app.include_router(walk_forward_router)
app.include_router(live_chart_router)


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
