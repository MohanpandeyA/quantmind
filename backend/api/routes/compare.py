"""Multi-ticker comparison endpoint.

WHY THIS FEATURE EXISTS:
    A trader often asks: "Which of these 5 stocks should I buy?"
    Instead of running 5 separate analyses and manually comparing,
    this endpoint runs all analyses in parallel and returns a ranked table.

    Example:
        POST /compare
        {"tickers": ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM"],
         "query": "Which is the best buy right now?"}

        Response:
        Rank 1: NVDA  | BUY  | Sharpe=0.84 | Return=+122.7%
        Rank 2: MSFT  | BUY  | Sharpe=0.71 | Return=+43.6%
        Rank 3: JPM   | BUY  | Sharpe=0.60 | Return=+30.8%
        Rank 4: GOOGL | HOLD | Sharpe=0.45 | Return=+18.2%
        Rank 5: AAPL  | HOLD | Sharpe=0.38 | Return=+16.3%

DESIGN PATTERN — Parallel execution with asyncio.gather():
    WHY: Running 5 analyses sequentially = 5 × 10s = 50s.
         Running them in parallel = max(10s, 10s, 10s, 10s, 10s) = 10s.
         asyncio.gather() runs all coroutines concurrently.

    CAVEAT: Yahoo Finance rate limits. We add a small stagger (0.5s between
    starts) to avoid hitting the rate limit simultaneously.

RANKING ALGORITHM:
    Composite score = 0.4 × Sharpe + 0.3 × Return + 0.2 × (1-MDD) + 0.1 × WinRate
    WHY these weights:
    - Sharpe (40%): Most important — risk-adjusted return
    - Return (30%): Raw performance matters
    - MDD (20%): Drawdown risk — lower is better
    - WinRate (10%): Consistency of profitable trades
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from config.logging_config import get_logger
from graph.workflow import run_analysis

logger = get_logger(__name__)

router = APIRouter(prefix="/compare", tags=["Comparison"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CompareRequest(BaseModel):
    """Request to compare multiple tickers.

    Attributes:
        tickers: List of ticker symbols to compare (2-10).
        query: Natural language question applied to all tickers.
        start_date: Backtest start date.
        end_date: Backtest end date.
    """

    tickers: List[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="List of 2-10 ticker symbols to compare",
    )
    query: str = Field(
        default="Which is the best investment right now?",
        min_length=5,
        max_length=500,
    )
    start_date: str = Field(default="2022-01-01")
    end_date: str = Field(default="2024-12-31")

    @field_validator("tickers")
    @classmethod
    def uppercase_tickers(cls, v: List[str]) -> List[str]:
        return [t.strip().upper() for t in v]


class TickerRanking(BaseModel):
    """Ranking result for a single ticker."""

    rank: int
    ticker: str
    signal: str
    composite_score: float
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    strategy: str
    summary: str  # One-line AI summary


class CompareResponse(BaseModel):
    """Response from the comparison endpoint."""

    rankings: List[TickerRanking]
    best_ticker: str
    worst_ticker: str
    query: str
    total_tickers: int
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=CompareResponse,
    summary="Compare multiple tickers and rank them",
    description="""
    Runs analysis for multiple tickers in parallel and returns a ranked table.

    Ranking formula:
        score = 0.4 × Sharpe + 0.3 × Return + 0.2 × (1-MDD) + 0.1 × WinRate

    Uses asyncio.gather() for parallel execution — all tickers analyzed
    simultaneously instead of sequentially.
    """,
)
async def compare_tickers(request: CompareRequest) -> CompareResponse:
    """Compare multiple tickers and rank by composite score.

    Args:
        request: CompareRequest with tickers list and query.

    Returns:
        CompareResponse with ranked tickers.
    """
    import time
    start_time = time.perf_counter()

    logger.info(
        "Compare | starting | tickers=%s | query=%r",
        request.tickers, request.query[:50],
    )

    # Run all analyses in parallel with small stagger to avoid rate limits
    # WHY stagger: Yahoo Finance blocks simultaneous requests from same IP.
    # 0.5s between starts = all finish at roughly the same time but don't
    # all hit Yahoo Finance at the exact same millisecond.
    async def analyze_with_delay(ticker: str, delay: float) -> Dict[str, Any]:
        await asyncio.sleep(delay)
        try:
            result = await run_analysis(
                ticker=ticker,
                query=request.query,
                start_date=request.start_date,
                end_date=request.end_date,
            )
            return {"ticker": ticker, "result": result, "error": None}
        except Exception as e:
            logger.warning("Compare | ticker failed | ticker=%s | %s", ticker, e)
            return {"ticker": ticker, "result": None, "error": str(e)}

    # Stagger starts by 0.5s each
    tasks = [
        analyze_with_delay(ticker, i * 0.5)
        for i, ticker in enumerate(request.tickers)
    ]
    results = await asyncio.gather(*tasks)

    # Build rankings
    rankings: List[TickerRanking] = []
    for item in results:
        ticker = item["ticker"]
        result = item["result"]

        if not result or item["error"]:
            # Failed analysis — put at bottom with zero scores
            rankings.append(TickerRanking(
                rank=0,
                ticker=ticker,
                signal="HOLD",
                composite_score=0.0,
                sharpe_ratio=0.0,
                total_return=0.0,
                max_drawdown=1.0,
                win_rate=0.0,
                strategy="unknown",
                summary=f"Analysis failed: {item.get('error', 'unknown error')}",
            ))
            continue

        bt = result.get("backtest_results", {})
        sharpe = bt.get("sharpe_ratio", 0.0)
        total_return = bt.get("total_return", 0.0)
        max_dd = bt.get("max_drawdown", 1.0)
        win_rate = bt.get("win_rate", 0.0)
        signal = result.get("signal", "HOLD")
        strategy = result.get("selected_strategy", "unknown")

        # Composite score formula
        # Normalize each metric to 0-1 range:
        # - Sharpe: cap at 3.0 (Sharpe > 3 is exceptional)
        # - Return: cap at 200% (200%+ return is exceptional)
        # - MDD: already 0-1 (lower is better → use 1-MDD)
        # - WinRate: already 0-1
        sharpe_norm = min(sharpe / 3.0, 1.0) if sharpe > 0 else 0.0
        return_norm = min(total_return / 2.0, 1.0) if total_return > 0 else 0.0
        mdd_norm = 1.0 - min(max_dd, 1.0)
        win_norm = win_rate

        composite = (
            0.40 * sharpe_norm
            + 0.30 * return_norm
            + 0.20 * mdd_norm
            + 0.10 * win_norm
        )

        # One-line summary from explanation
        explanation = result.get("final_explanation", "")
        summary_lines = [l for l in explanation.split("\n") if l.strip() and not l.startswith("SIGNAL")]
        summary = summary_lines[0][:100] if summary_lines else f"{ticker}: {signal}"

        rankings.append(TickerRanking(
            rank=0,  # Will be set after sorting
            ticker=ticker,
            signal=signal,
            composite_score=round(composite, 4),
            sharpe_ratio=round(sharpe, 3),
            total_return=round(total_return, 4),
            max_drawdown=round(max_dd, 4),
            win_rate=round(win_rate, 3),
            strategy=strategy,
            summary=summary,
        ))

    # Sort by composite score (highest first) and assign ranks
    rankings.sort(key=lambda r: r.composite_score, reverse=True)
    for i, r in enumerate(rankings):
        r.rank = i + 1

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Compare | complete | tickers=%d | best=%s | time=%.0fms",
        len(rankings),
        rankings[0].ticker if rankings else "none",
        duration_ms,
    )

    return CompareResponse(
        rankings=rankings,
        best_ticker=rankings[0].ticker if rankings else "",
        worst_ticker=rankings[-1].ticker if rankings else "",
        query=request.query,
        total_tickers=len(rankings),
        processing_time_ms=round(duration_ms, 1),
    )
