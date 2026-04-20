"""Pydantic request/response schemas for the QuantMind FastAPI.

All API inputs and outputs are validated by Pydantic models.
This ensures:
- Type safety: wrong types rejected at the API boundary
- Auto-documentation: FastAPI generates OpenAPI docs from these models
- Serialization: clean JSON responses with proper types
- Validation: business rules enforced (e.g., valid date format)
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    """Request body for POST /analyze.

    Attributes:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'RELIANCE.NS').
        query: Natural language question about the stock.
        start_date: Backtest start date (YYYY-MM-DD). Default 2 years ago.
        end_date: Backtest end date (YYYY-MM-DD). Default today.

    Example:
        {
            "ticker": "AAPL",
            "query": "Should I buy Apple stock given the current market conditions?",
            "start_date": "2022-01-01",
            "end_date": "2024-12-31"
        }
    """

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'RELIANCE.NS')",
        examples=["AAPL"],
    )
    query: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Natural language question about the stock",
        examples=["Should I buy Apple stock given the current market conditions?"],
    )
    start_date: str = Field(
        default="2022-01-01",
        description="Backtest start date (YYYY-MM-DD)",
        examples=["2022-01-01"],
    )
    end_date: str = Field(
        default="2024-12-31",
        description="Backtest end date (YYYY-MM-DD)",
        examples=["2024-12-31"],
    )

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        """Normalize ticker to uppercase."""
        return v.strip().upper()

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date is in YYYY-MM-DD format."""
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v!r}")
        return v

    def model_post_init(self, __context: object) -> None:
        """Validate that start_date is before end_date.

        WHY: yfinance raises 'Invalid input - start date cannot be after end date'
        if start_date >= end_date. Catching this at the API boundary gives a
        clear error message instead of a cryptic yfinance error.
        """
        try:
            start = date.fromisoformat(self.start_date)
            end = date.fromisoformat(self.end_date)
            if start >= end:
                raise ValueError(
                    f"start_date ({self.start_date}) must be before "
                    f"end_date ({self.end_date})."
                )
        except ValueError as e:
            if "must be before" in str(e):
                raise

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Ensure query has meaningful content."""
        if not v.strip():
            raise ValueError("query must not be empty or whitespace.")
        return v.strip()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class MarketDataResponse(BaseModel):
    """Market data in the analysis response."""

    ticker: str
    current_price: float
    price_change_pct: float
    volume: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    week_52_high: float
    week_52_low: float
    avg_volume: float


class BacktestResultsResponse(BaseModel):
    """Backtest performance metrics in the analysis response."""

    strategy_name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    win_rate: float
    profit_factor: float
    n_trades: int
    n_days: int
    start_date: str
    end_date: str


class RiskMetricsResponse(BaseModel):
    """Risk assessment in the analysis response."""

    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    risk_score: float
    risk_level: str
    risk_approved: bool
    rejection_reason: str = ""


class AnalysisResponse(BaseModel):
    """Full response from POST /analyze.

    This is the complete output of the LangGraph workflow,
    formatted for the frontend dashboard.

    Attributes:
        ticker: Stock ticker analyzed.
        query: Original user query.
        signal: Trading signal (BUY/SELL/HOLD).
        final_explanation: LLM-generated cited explanation.
        final_citations: Source citation strings.
        market_data: Current market data.
        selected_strategy: Strategy used (momentum/mean_reversion).
        strategy_rationale: Why this strategy was selected.
        backtest_results: Full backtest performance metrics.
        equity_curve: Portfolio value over time (for charting).
        risk_metrics: Risk assessment results.
        processing_time_ms: Total analysis time in milliseconds.
        error: Error message if any agent failed (empty if success).
    """

    ticker: str
    query: str
    signal: str = Field(description="BUY, SELL, or HOLD")
    final_explanation: str
    final_citations: List[str] = Field(default_factory=list)
    market_data: Optional[MarketDataResponse] = None
    selected_strategy: str = ""
    strategy_rationale: str = ""
    backtest_results: Optional[BacktestResultsResponse] = None
    equity_curve: List[float] = Field(default_factory=list)
    risk_metrics: Optional[RiskMetricsResponse] = None
    # Sentiment analysis (SentimentAgent — 7th LangGraph node)
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    sentiment_confidence: Optional[float] = None
    sentiment_details: Optional[Dict[str, Any]] = None
    processing_time_ms: float = 0.0
    error: str = ""

    @classmethod
    def from_trading_state(cls, state: Dict[str, Any]) -> "AnalysisResponse":
        """Build AnalysisResponse from a TradingState dict.

        Args:
            state: Final TradingState from the LangGraph workflow.

        Returns:
            AnalysisResponse ready to serialize as JSON.
        """
        # Market data
        md = state.get("market_data")
        market_data_resp = MarketDataResponse(**md) if md else None

        # Backtest results
        bt = state.get("backtest_results")
        backtest_resp = BacktestResultsResponse(**bt) if bt else None

        # Risk metrics
        rm = state.get("risk_metrics")
        risk_resp = RiskMetricsResponse(**rm) if rm else None

        return cls(
            ticker=state.get("ticker", ""),
            query=state.get("query", ""),
            signal=state.get("signal", "HOLD"),
            final_explanation=state.get("final_explanation", ""),
            final_citations=state.get("final_citations", []),
            market_data=market_data_resp,
            selected_strategy=state.get("selected_strategy", ""),
            strategy_rationale=state.get("strategy_rationale", ""),
            backtest_results=backtest_resp,
            equity_curve=state.get("equity_curve", [])[:252],  # Last year
            risk_metrics=risk_resp,
            sentiment_score=state.get("sentiment_score"),
            sentiment_label=state.get("sentiment_label"),
            sentiment_confidence=state.get("sentiment_confidence"),
            sentiment_details=state.get("sentiment_details"),
            processing_time_ms=state.get("processing_time_ms", 0.0),
            error=state.get("error", ""),
        )


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str = "ok"
    version: str = "1.0.0"
    phase: str = "Phase 3 — LangGraph Agents"


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str = ""
    ticker: str = ""
