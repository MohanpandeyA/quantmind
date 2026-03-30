"""Analysis route — POST /analyze endpoint.

The main API endpoint that triggers the full LangGraph workflow:
    ResearchAgent → RAGAgent → StrategyAgent → BacktestAgent
    → RiskAgent → ExplainerAgent

Returns a complete trading analysis with:
- BUY/SELL/HOLD signal
- LLM-generated explanation with source citations
- Backtest performance metrics
- Risk assessment
- Equity curve for charting
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from api.schemas import AnalysisRequest, AnalysisResponse, ErrorResponse
from config.logging_config import get_logger
from graph.workflow import run_analysis

logger = get_logger(__name__)

router = APIRouter(prefix="/analyze", tags=["Analysis"])


@router.post(
    "",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Run full trading analysis",
    description="""
    Runs the complete QuantMind analysis pipeline:
    1. **ResearchAgent**: Fetches real-time market data (yfinance)
    2. **RAGAgent**: Retrieves relevant SEC filings and news (ChromaDB)
    3. **StrategyAgent**: Selects optimal strategy (Momentum or MeanReversion)
    4. **BacktestAgent**: Runs historical backtest on real data
    5. **RiskAgent**: Evaluates risk limits (Sharpe, VaR, Drawdown)
    6. **ExplainerAgent**: Generates cited explanation (Groq LLM)

    Returns a BUY/SELL/HOLD signal with full justification and source citations.
    """,
    responses={
        200: {"description": "Analysis complete"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Analysis failed"},
    },
)
async def analyze(request: AnalysisRequest) -> AnalysisResponse:
    """Run full trading analysis for a ticker.

    Args:
        request: AnalysisRequest with ticker, query, and date range.

    Returns:
        AnalysisResponse with signal, explanation, metrics, and citations.

    Raises:
        HTTPException 400: If ticker is invalid.
        HTTPException 500: If the workflow fails unexpectedly.

    Example request:
        POST /analyze
        {
            "ticker": "AAPL",
            "query": "Should I buy Apple stock?",
            "start_date": "2022-01-01",
            "end_date": "2024-12-31"
        }

    Example response:
        {
            "ticker": "AAPL",
            "signal": "BUY",
            "final_explanation": "SIGNAL: BUY\\nCONFIDENCE: MEDIUM\\n...",
            "backtest_results": {"sharpe_ratio": 1.42, ...},
            "processing_time_ms": 8340.5
        }
    """
    logger.info(
        "POST /analyze | ticker=%s | query=%r",
        request.ticker, request.query[:50],
    )

    try:
        # Run the full LangGraph workflow
        final_state = await run_analysis(
            ticker=request.ticker,
            query=request.query,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        # Check for critical errors
        error = final_state.get("error", "")
        if error and not final_state.get("final_explanation"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error,
            )

        # Build and return response
        response = AnalysisResponse.from_trading_state(final_state)

        logger.info(
            "POST /analyze | complete | ticker=%s | signal=%s | time=%.0fms",
            request.ticker,
            response.signal,
            response.processing_time_ms,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "POST /analyze | unexpected error | ticker=%s | %s",
            request.ticker, e, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {e}",
        )
