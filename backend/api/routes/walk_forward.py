"""Walk-Forward Validation endpoint.

POST /walk-forward
    Runs walk-forward validation to test if a strategy is robust
    (not overfitted to historical data).

    For each rolling window:
        1. Optimize parameters on TRAIN data
        2. Test those parameters on unseen TEST data

    Returns in-sample vs out-of-sample comparison + robustness verdict.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from config.logging_config import get_logger
from engine.walk_forward import run_walk_forward

logger = get_logger(__name__)

router = APIRouter(prefix="/walk-forward", tags=["Walk-Forward Validation"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class WalkForwardRequest(BaseModel):
    """Request body for POST /walk-forward."""

    ticker: str = Field(..., min_length=1, max_length=20)
    strategy: str = Field(default="momentum")
    start_date: str = Field(default="2022-01-01")
    end_date: str = Field(default="2024-12-31")
    train_months: int = Field(default=12, ge=3, le=24)
    test_months: int = Field(default=3, ge=1, le=12)
    step_months: int = Field(default=3, ge=1, le=6)
    optimize_for: str = Field(default="sharpe")

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid = ("momentum", "mean_reversion", "rsi", "macd")
        if v not in valid:
            raise ValueError(f"strategy must be one of {valid}")
        return v

    @field_validator("optimize_for")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        valid = ("sharpe", "total_return", "calmar")
        if v not in valid:
            raise ValueError(f"optimize_for must be one of {valid}")
        return v


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("")
async def walk_forward_validation(request: WalkForwardRequest) -> JSONResponse:
    """Run walk-forward validation for a strategy.

    Tests whether a strategy is robust (not overfitted) by comparing
    in-sample (training) performance to out-of-sample (test) performance.

    Args:
        request: WalkForwardRequest with ticker, strategy, and window config.

    Returns:
        JSON with in-sample vs out-of-sample comparison and robustness verdict.
    """
    logger.info(
        "WalkForward | starting | ticker=%s | strategy=%s | "
        "train=%dmo | test=%dmo",
        request.ticker, request.strategy,
        request.train_months, request.test_months,
    )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            run_walk_forward,
            request.ticker,
            request.strategy,
            request.start_date,
            request.end_date,
            request.train_months,
            request.test_months,
            request.step_months,
            request.optimize_for,
        )

        # Serialize result (sanitize inf/nan)
        response_dict = _sanitize(result.__dict__)

        # Serialize window results
        response_dict["windows"] = [
            _sanitize(w.__dict__) for w in result.windows
        ]

        logger.info(
            "WalkForward | complete | ticker=%s | in_sample=%.2f | "
            "out_of_sample=%.2f | verdict=%s | time=%.0fms",
            request.ticker,
            result.in_sample_sharpe,
            result.out_of_sample_sharpe,
            result.verdict,
            result.processing_time_ms,
        )

        return JSONResponse(content=response_dict)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            "WalkForward | failed | ticker=%s | %s",
            request.ticker, e, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Walk-forward validation failed: {e}",
        )


def _sanitize(obj: Any) -> Any:
    """Recursively replace inf/nan with None for JSON compliance."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj
