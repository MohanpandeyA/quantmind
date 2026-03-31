"""Portfolio API routes — real-time P&L and position management.

WHY THIS ENDPOINT EXISTS:
    A trader manages multiple positions simultaneously. This endpoint:
    1. Fetches current prices for all positions via yfinance (free)
    2. Calculates real-time P&L (unrealized gain/loss) for each position
    3. Computes portfolio-level metrics (total return, best/worst position)
    4. Returns everything in one API call — no N+1 queries

DESIGN PATTERN — Batch price fetching:
    Instead of calling yfinance once per position (N calls for N positions),
    we use yfinance.download() with multiple tickers in one call.
    This is the 'batch API' pattern — dramatically reduces latency.

    N=5 positions:
    - Naive: 5 × yfinance calls × ~1s each = 5s
    - Batch: 1 × yfinance call with 5 tickers = ~1.5s
    - Speedup: 3.3× faster

Routes:
    GET  /portfolio/performance  → Real-time P&L for all positions
    POST /portfolio/positions    → Add a new position
    DELETE /portfolio/positions/:ticker → Remove a position
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])

# In-memory portfolio storage (no MongoDB needed for demo)
# WHY in-memory: Simpler for demo. When MongoDB is configured,
# this can be replaced with Mongoose queries without changing the API.
_portfolio: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class AddPositionRequest(BaseModel):
    """Request to add a position to the portfolio.

    Attributes:
        ticker: Stock ticker symbol.
        shares: Number of shares held.
        entry_price: Price paid per share (USD).
        entry_date: Date position was opened (YYYY-MM-DD).
        notes: Optional trader notes.
    """

    ticker: str = Field(..., min_length=1, max_length=20)
    shares: float = Field(..., gt=0, description="Number of shares (must be positive)")
    entry_price: float = Field(..., gt=0, description="Entry price per share in USD")
    entry_date: str = Field(..., description="Entry date YYYY-MM-DD")
    notes: str = Field(default="", max_length=500)

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("entry_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"entry_date must be YYYY-MM-DD, got: {v!r}")
        return v


class PositionPerformance(BaseModel):
    """Real-time performance for a single position."""

    ticker: str
    shares: float
    entry_price: float
    entry_date: str
    notes: str
    current_price: float
    price_change_pct: float       # Today's % change
    cost_basis: float             # shares × entry_price
    current_value: float          # shares × current_price
    unrealized_pnl: float         # current_value - cost_basis
    unrealized_pnl_pct: float     # unrealized_pnl / cost_basis × 100
    week_52_high: float
    week_52_low: float


class PortfolioPerformanceResponse(BaseModel):
    """Full portfolio performance response."""

    positions: List[PositionPerformance]
    total_cost_basis: float
    total_current_value: float
    total_unrealized_pnl: float
    total_unrealized_pnl_pct: float
    best_performer: Optional[str]   # Ticker with highest unrealized_pnl_pct
    worst_performer: Optional[str]  # Ticker with lowest unrealized_pnl_pct
    position_count: int
    last_updated: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/performance",
    response_model=PortfolioPerformanceResponse,
    summary="Get real-time portfolio performance",
    description="""
    Fetches current prices for all portfolio positions using yfinance (free).
    Calculates unrealized P&L for each position and portfolio totals.

    Uses batch price fetching — all tickers in one yfinance call (3× faster
    than fetching each ticker individually).
    """,
)
async def get_portfolio_performance() -> PortfolioPerformanceResponse:
    """Get real-time P&L for all portfolio positions.

    Returns:
        PortfolioPerformanceResponse with per-position and portfolio totals.
    """
    if not _portfolio:
        return PortfolioPerformanceResponse(
            positions=[],
            total_cost_basis=0.0,
            total_current_value=0.0,
            total_unrealized_pnl=0.0,
            total_unrealized_pnl_pct=0.0,
            best_performer=None,
            worst_performer=None,
            position_count=0,
            last_updated=datetime.utcnow().isoformat(),
        )

    # Batch fetch current prices for all tickers
    tickers = list({p["ticker"] for p in _portfolio})
    prices = await _fetch_current_prices(tickers)

    # Calculate per-position performance
    positions: List[PositionPerformance] = []
    for pos in _portfolio:
        ticker = pos["ticker"]
        price_data = prices.get(ticker, {})
        current_price = price_data.get("current_price", pos["entry_price"])
        price_change_pct = price_data.get("price_change_pct", 0.0)
        week_52_high = price_data.get("week_52_high", current_price)
        week_52_low = price_data.get("week_52_low", current_price)

        cost_basis = pos["shares"] * pos["entry_price"]
        current_value = pos["shares"] * current_price
        unrealized_pnl = current_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0

        positions.append(PositionPerformance(
            ticker=ticker,
            shares=pos["shares"],
            entry_price=pos["entry_price"],
            entry_date=pos["entry_date"],
            notes=pos.get("notes", ""),
            current_price=round(current_price, 2),
            price_change_pct=round(price_change_pct, 2),
            cost_basis=round(cost_basis, 2),
            current_value=round(current_value, 2),
            unrealized_pnl=round(unrealized_pnl, 2),
            unrealized_pnl_pct=round(unrealized_pnl_pct, 2),
            week_52_high=round(week_52_high, 2),
            week_52_low=round(week_52_low, 2),
        ))

    # Portfolio totals
    total_cost = sum(p.cost_basis for p in positions)
    total_value = sum(p.current_value for p in positions)
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

    # Best/worst performers
    best = max(positions, key=lambda p: p.unrealized_pnl_pct).ticker if positions else None
    worst = min(positions, key=lambda p: p.unrealized_pnl_pct).ticker if positions else None

    logger.info(
        "Portfolio performance | positions=%d | total_value=%.2f | pnl=%.2f%%",
        len(positions), total_value, total_pnl_pct,
    )

    return PortfolioPerformanceResponse(
        positions=positions,
        total_cost_basis=round(total_cost, 2),
        total_current_value=round(total_value, 2),
        total_unrealized_pnl=round(total_pnl, 2),
        total_unrealized_pnl_pct=round(total_pnl_pct, 2),
        best_performer=best,
        worst_performer=worst,
        position_count=len(positions),
        last_updated=datetime.utcnow().isoformat(),
    )


@router.post(
    "/positions",
    status_code=status.HTTP_201_CREATED,
    summary="Add a position to the portfolio",
)
async def add_position(request: AddPositionRequest) -> Dict[str, Any]:
    """Add a new stock position to the portfolio.

    Args:
        request: Position details (ticker, shares, entry_price, entry_date).

    Returns:
        The added position with calculated cost basis.
    """
    # Check if ticker already exists — update shares instead of duplicate
    for pos in _portfolio:
        if pos["ticker"] == request.ticker:
            # Average down/up: recalculate average entry price
            total_shares = pos["shares"] + request.shares
            avg_price = (
                (pos["shares"] * pos["entry_price"] + request.shares * request.entry_price)
                / total_shares
            )
            pos["shares"] = total_shares
            pos["entry_price"] = round(avg_price, 4)
            pos["notes"] = request.notes or pos["notes"]
            logger.info(
                "Portfolio | position updated | ticker=%s | shares=%.2f | avg_price=%.2f",
                request.ticker, total_shares, avg_price,
            )
            return {
                "message": f"Position updated (averaged) | {request.ticker}",
                "position": pos,
            }

    # New position
    position = {
        "ticker": request.ticker,
        "shares": request.shares,
        "entry_price": request.entry_price,
        "entry_date": request.entry_date,
        "notes": request.notes,
        "cost_basis": round(request.shares * request.entry_price, 2),
    }
    _portfolio.append(position)

    logger.info(
        "Portfolio | position added | ticker=%s | shares=%.2f | entry=%.2f | cost=%.2f",
        request.ticker, request.shares, request.entry_price, position["cost_basis"],
    )

    return {
        "message": f"Position added | {request.ticker}",
        "position": position,
        "portfolio_size": len(_portfolio),
    }


@router.delete(
    "/positions/{ticker}",
    summary="Remove a position from the portfolio",
)
async def remove_position(ticker: str) -> Dict[str, str]:
    """Remove a position from the portfolio.

    Args:
        ticker: Ticker symbol to remove.

    Returns:
        Confirmation message.

    Raises:
        HTTPException 404: If ticker not in portfolio.
    """
    ticker = ticker.upper()
    global _portfolio
    original_len = len(_portfolio)
    _portfolio = [p for p in _portfolio if p["ticker"] != ticker]

    if len(_portfolio) == original_len:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker '{ticker}' not found in portfolio.",
        )

    logger.info("Portfolio | position removed | ticker=%s", ticker)
    return {"message": f"Position removed | {ticker}"}


@router.get(
    "/positions",
    summary="List all portfolio positions",
)
async def list_positions() -> Dict[str, Any]:
    """List all positions in the portfolio.

    Returns:
        List of positions with cost basis.
    """
    return {
        "positions": _portfolio,
        "count": len(_portfolio),
        "total_cost_basis": round(
            sum(p["shares"] * p["entry_price"] for p in _portfolio), 2
        ),
    }


# ---------------------------------------------------------------------------
# Helper: batch price fetching
# ---------------------------------------------------------------------------

async def _fetch_current_prices(tickers: List[str]) -> Dict[str, Dict[str, float]]:
    """Fetch current prices for multiple tickers in one yfinance call.

    WHY BATCH FETCHING:
        yfinance.download() accepts a list of tickers and fetches all
        in a single HTTP request. This is 3-5× faster than fetching
        each ticker individually.

        N=5 tickers:
        - Individual: 5 calls × ~1s = 5s
        - Batch: 1 call = ~1.5s

    Args:
        tickers: List of ticker symbols.

    Returns:
        Dict mapping ticker → {current_price, price_change_pct, week_52_high, week_52_low}
    """
    import asyncio
    import warnings
    warnings.filterwarnings("ignore")

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_prices_sync, tickers)


def _fetch_prices_sync(tickers: List[str]) -> Dict[str, Dict[str, float]]:
    """Synchronous price fetching — runs in thread pool.

    Args:
        tickers: List of ticker symbols.

    Returns:
        Price data dict.
    """
    import warnings
    warnings.filterwarnings("ignore")

    try:
        import yfinance as yf

        # Batch download: one call for all tickers
        # period="1y" gives us 52-week high/low
        data = yf.download(
            tickers,
            period="1y",
            auto_adjust=True,
            progress=False,
            group_by="ticker" if len(tickers) > 1 else "column",
        )

        result: Dict[str, Dict[str, float]] = {}

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    hist = data
                else:
                    hist = data[ticker] if ticker in data.columns.get_level_values(0) else data

                if hist.empty:
                    continue

                # Handle MultiIndex columns from yfinance 1.x
                if hasattr(hist.columns, "levels"):
                    hist.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                                    for c in hist.columns]
                else:
                    hist.columns = [c.lower() for c in hist.columns]

                current = float(hist["close"].iloc[-1])
                prev = float(hist["close"].iloc[-2]) if len(hist) > 1 else current
                change_pct = ((current - prev) / prev * 100) if prev > 0 else 0.0

                result[ticker] = {
                    "current_price": current,
                    "price_change_pct": change_pct,
                    "week_52_high": float(hist["high"].max()),
                    "week_52_low": float(hist["low"].min()),
                }
            except Exception as e:
                logger.debug("Price fetch failed for %s: %s", ticker, e)

        return result

    except Exception as e:
        logger.error("Batch price fetch failed: %s", e)
        return {}
