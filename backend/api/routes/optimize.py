"""Strategy Parameter Optimizer endpoint.

WHY THIS FEATURE EXISTS:
    The default strategy parameters (short_window=20, long_window=50) work
    reasonably well, but they're not optimal for every stock. NVDA behaves
    very differently from JPM — different volatility, trend strength, cycle length.

    This endpoint runs a grid search over parameter combinations and finds
    the parameters that maximize the Sharpe ratio for a specific ticker.

    Example:
        POST /optimize
        {"ticker": "AAPL", "strategy": "momentum", "optimize_for": "sharpe"}

        Response:
        Best params: {"short_window": 10, "long_window": 30}
        Best Sharpe: 1.42 (vs default 0.38)
        Improvement: +273%

DESIGN PATTERN — Grid Search:
    Test all combinations of parameters, evaluate each, return the best.
    This is the simplest hyperparameter optimization method.

    For momentum: 3 short × 4 long = 12 combinations
    For mean_reversion: 3 windows × 4 z_thresholds = 12 combinations

    WHY not Bayesian optimization:
    - Grid search is simple, interpretable, and sufficient for 12 combinations
    - Bayesian optimization is overkill for this parameter space
    - Traders can understand "I tested 12 combinations" easily

OPTIMIZATION METRIC OPTIONS:
    - sharpe: Risk-adjusted return (recommended)
    - total_return: Raw return (ignores risk)
    - calmar: Return / max drawdown (good for risk-averse traders)

IMPORTANT: This uses the data cache from Critical Fix 1.
    All 12 backtests reuse the same downloaded DataFrame — only 1 yfinance call.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from config.logging_config import get_logger
from engine.backtester import Backtester, BacktestConfig
from engine.strategies.base_strategy import StrategyConfig
from engine.strategies.macd_strategy import MACDStrategy
from engine.strategies.mean_reversion import MeanReversionStrategy
from engine.strategies.momentum import MomentumStrategy
from engine.strategies.rsi_strategy import RSIStrategy

logger = get_logger(__name__)

router = APIRouter(prefix="/optimize", tags=["Optimization"])

# Parameter grids to search
MOMENTUM_GRID = {
    "short_window": [5, 10, 20],
    "long_window": [20, 30, 50, 100],
}

MEAN_REVERSION_GRID = {
    "window": [10, 20, 30],
    "z_threshold": [1.0, 1.5, 2.0, 2.5],
}

RSI_GRID = {
    "period": [7, 14, 21],
    "oversold": [25, 30, 35],
    "overbought": [65, 70, 75],
}

MACD_GRID = {
    "fast": [8, 12, 16],
    "slow": [21, 26, 30],
    "signal_period": [7, 9, 12],
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    """Request to optimize strategy parameters.

    Attributes:
        ticker: Stock ticker to optimize for.
        strategy: Strategy type ('momentum' or 'mean_reversion').
        optimize_for: Metric to maximize ('sharpe', 'total_return', 'calmar').
        start_date: Backtest start date.
        end_date: Backtest end date.
    """

    ticker: str = Field(..., min_length=1, max_length=20)
    strategy: str = Field(
        default="momentum",
        description="Strategy to optimize: 'momentum' or 'mean_reversion'",
    )
    optimize_for: str = Field(
        default="sharpe",
        description="Metric to maximize: 'sharpe', 'total_return', or 'calmar'",
    )
    start_date: str = Field(default="2022-01-01")
    end_date: str = Field(default="2024-12-31")

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        if v not in ("momentum", "mean_reversion", "rsi", "macd"):
            raise ValueError("strategy must be 'momentum', 'mean_reversion', 'rsi', or 'macd'")
        return v

    @field_validator("optimize_for")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        if v not in ("sharpe", "total_return", "calmar"):
            raise ValueError("optimize_for must be 'sharpe', 'total_return', or 'calmar'")
        return v


class ParamResult(BaseModel):
    """Result for a single parameter combination."""

    params: Dict[str, Any]
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    n_trades: int
    score: float  # The metric being optimized


class OptimizeResponse(BaseModel):
    """Response from the optimization endpoint."""

    ticker: str
    strategy: str
    optimize_for: str
    best_params: Dict[str, Any]
    best_score: float
    default_score: float
    improvement_pct: float
    all_results: List[ParamResult]
    total_combinations_tested: int
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=OptimizeResponse,
    summary="Find optimal strategy parameters via grid search",
    description="""
    Tests all parameter combinations and returns the best for a given ticker.

    Uses the data cache (Critical Fix 1) — downloads price data only ONCE,
    then runs all backtests on the cached DataFrame.

    Momentum grid: 3 short_window × 4 long_window = 12 combinations
    MeanReversion grid: 3 window × 4 z_threshold = 12 combinations
    """,
)
async def optimize_strategy(request: OptimizeRequest) -> OptimizeResponse:
    """Find optimal strategy parameters for a ticker.

    Args:
        request: OptimizeRequest with ticker, strategy, and metric.

    Returns:
        OptimizeResponse with best params and all tested combinations.
    """
    import time
    start_time = time.perf_counter()

    logger.info(
        "Optimize | starting | ticker=%s | strategy=%s | metric=%s",
        request.ticker, request.strategy, request.optimize_for,
    )

    # Run grid search in thread pool (CPU-bound backtesting)
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        _run_grid_search,
        request.ticker,
        request.strategy,
        request.optimize_for,
        request.start_date,
        request.end_date,
    )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grid search failed for {request.ticker}",
        )

    # Sort by score (highest first)
    results.sort(key=lambda r: r.score, reverse=True)
    best = results[0]

    # Get default params score for comparison
    default_params = _get_default_params(request.strategy)
    default_result = next(
        (r for r in results if r.params == default_params), None
    )
    default_score = default_result.score if default_result else 0.0

    improvement = (
        ((best.score - default_score) / abs(default_score) * 100)
        if default_score != 0 else 0.0
    )

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Optimize | complete | ticker=%s | best_params=%s | best_score=%.3f | "
        "improvement=%.1f%% | time=%.0fms",
        request.ticker, best.params, best.score, improvement, duration_ms,
    )

    return OptimizeResponse(
        ticker=request.ticker,
        strategy=request.strategy,
        optimize_for=request.optimize_for,
        best_params=best.params,
        best_score=round(best.score, 4),
        default_score=round(default_score, 4),
        improvement_pct=round(improvement, 1),
        all_results=results,
        total_combinations_tested=len(results),
        processing_time_ms=round(duration_ms, 1),
    )


# ---------------------------------------------------------------------------
# Grid search (synchronous — runs in thread pool)
# ---------------------------------------------------------------------------

def _run_grid_search(
    ticker: str,
    strategy_name: str,
    optimize_for: str,
    start_date: str,
    end_date: str,
) -> List[ParamResult]:
    """Run grid search over all parameter combinations.

    WHY synchronous: Backtesting is CPU-bound (numpy operations).
    Running in asyncio event loop would block other requests.
    We use run_in_executor() to run this in a thread pool.

    USES DATA CACHE: The first backtest downloads the DataFrame.
    All subsequent backtests reuse the cached DataFrame (Critical Fix 1).
    This means 12 backtests = 1 yfinance download + 11 cache hits.

    Args:
        ticker: Stock ticker.
        strategy_name: 'momentum' or 'mean_reversion'.
        optimize_for: Metric to maximize.
        start_date: Backtest start date.
        end_date: Backtest end date.

    Returns:
        List of ParamResult for each parameter combination.
    """
    grid = {
        "momentum": MOMENTUM_GRID,
        "mean_reversion": MEAN_REVERSION_GRID,
        "rsi": RSI_GRID,
        "macd": MACD_GRID,
    }.get(strategy_name, MOMENTUM_GRID)

    results: List[ParamResult] = []

    # Generate all parameter combinations
    param_combinations = _generate_combinations(grid)

    for params in param_combinations:
        try:
            # Skip invalid combinations
            if strategy_name == "momentum":
                if params.get("short_window", 0) >= params.get("long_window", 0):
                    continue
            if strategy_name == "macd":
                if params.get("fast", 0) >= params.get("slow", 0):
                    continue
            if strategy_name == "rsi":
                if params.get("oversold", 0) >= params.get("overbought", 100):
                    continue

            config = StrategyConfig(params=params)

            if strategy_name == "momentum":
                strategy = MomentumStrategy(config)
            elif strategy_name == "mean_reversion":
                strategy = MeanReversionStrategy(config)
            elif strategy_name == "rsi":
                strategy = RSIStrategy(config)
            elif strategy_name == "macd":
                strategy = MACDStrategy(config)
            else:
                strategy = MomentumStrategy(config)

            bt_config = BacktestConfig(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                long_only=True,
                use_segment_tree=True,
            )

            backtester = Backtester(bt_config, strategy)
            result, report = backtester.run()

            # Get the optimization score
            score = _get_score(report, optimize_for)

            results.append(ParamResult(
                params=params,
                sharpe_ratio=round(report.sharpe_ratio, 3),
                total_return=round(report.total_return, 4),
                max_drawdown=round(report.max_drawdown, 4),
                calmar_ratio=round(report.calmar_ratio, 3),
                win_rate=round(report.win_rate, 3),
                n_trades=report.n_trades,
                score=round(score, 4),
            ))

        except Exception as e:
            logger.debug("Optimize | param failed | params=%s | %s", params, e)

    return results


def _generate_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations from a parameter grid.

    Args:
        grid: Dict mapping param_name → list of values.

    Returns:
        List of dicts, each representing one parameter combination.

    Example:
        grid = {"a": [1, 2], "b": [10, 20]}
        → [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20}]
    """
    import itertools
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations


def _get_score(report: object, optimize_for: str) -> float:
    """Extract the optimization score from a PerformanceReport.

    Args:
        report: PerformanceReport from backtester.
        optimize_for: Metric name.

    Returns:
        Score value (higher is better).
    """
    if optimize_for == "sharpe":
        return getattr(report, "sharpe_ratio", 0.0)
    elif optimize_for == "total_return":
        return getattr(report, "total_return", 0.0)
    elif optimize_for == "calmar":
        return getattr(report, "calmar_ratio", 0.0)
    return 0.0


def _get_default_params(strategy_name: str) -> Dict[str, Any]:
    """Return the default parameters for a strategy.

    Args:
        strategy_name: Strategy name.

    Returns:
        Default parameter dict.
    """
    if strategy_name == "momentum":
        return {"short_window": 20, "long_window": 50}
    else:
        return {"window": 20, "z_threshold": 2.0}
