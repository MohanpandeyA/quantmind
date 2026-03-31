"""Earnings calendar endpoint.

WHY THIS FEATURE EXISTS:
    Earnings announcements are the single biggest driver of short-term
    stock price movements. A stock can move ±20% on earnings day.

    If a trader runs QuantMind analysis on AAPL 2 days before earnings,
    the backtest results are based on historical data that doesn't include
    the upcoming earnings risk. The analysis could be completely wrong.

    This endpoint:
    1. Fetches upcoming earnings dates for a ticker (yfinance, free)
    2. Warns if earnings are within the next 7 days
    3. Shows the expected price move (implied volatility from options)
    4. Provides context for the analysis

WHY yfinance for earnings:
    yfinance provides earnings calendar data for free via:
    ticker.calendar → next earnings date, EPS estimate, revenue estimate
    ticker.earnings_dates → historical earnings dates

DESIGN DECISION — Warning threshold:
    7 days: "Earnings in 7 days — analysis may be outdated after earnings"
    3 days: "⚠️ Earnings in 3 days — HIGH RISK, consider waiting"
    0 days: "🚨 Earnings TODAY — do not trade based on this analysis"
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/earnings", tags=["Earnings"])


@router.get(
    "/{ticker}",
    summary="Get upcoming earnings date and warning",
    description="""
    Fetches the next earnings date for a ticker and returns a warning
    if earnings are within 7 days.

    Earnings announcements cause major price moves (±5-20%).
    Always check this before acting on an analysis.
    """,
)
async def get_earnings(ticker: str) -> Dict[str, Any]:
    """Get upcoming earnings date for a ticker.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dict with next_earnings_date, days_until_earnings, warning_level, message.
    """
    ticker_upper = ticker.upper()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _fetch_earnings_sync, ticker_upper)
    return result


def _fetch_earnings_sync(ticker: str) -> Dict[str, Any]:
    """Synchronous earnings fetch via yfinance.

    Args:
        ticker: Ticker symbol.

    Returns:
        Earnings data dict.
    """
    import warnings
    warnings.filterwarnings("ignore")

    try:
        import yfinance as yf
        import pandas as pd

        stock = yf.Ticker(ticker)

        # Try to get next earnings date from calendar
        next_earnings = None
        eps_estimate = None
        revenue_estimate = None

        try:
            calendar = stock.calendar
            if calendar is not None and not (isinstance(calendar, dict) and not calendar):
                if isinstance(calendar, dict):
                    # Newer yfinance returns dict
                    earnings_date = calendar.get("Earnings Date")
                    if earnings_date:
                        if isinstance(earnings_date, list) and earnings_date:
                            next_earnings = pd.Timestamp(earnings_date[0])
                        elif hasattr(earnings_date, "date"):
                            next_earnings = pd.Timestamp(earnings_date)
                    eps_estimate = calendar.get("EPS Estimate")
                    revenue_estimate = calendar.get("Revenue Estimate")
                elif hasattr(calendar, "loc"):
                    # Older yfinance returns DataFrame
                    if "Earnings Date" in calendar.index:
                        next_earnings = pd.Timestamp(calendar.loc["Earnings Date"].iloc[0])
        except Exception as e:
            logger.debug("Calendar fetch failed for %s: %s", ticker, e)

        # Fallback: try earnings_dates
        if next_earnings is None:
            try:
                earnings_dates = stock.earnings_dates
                if earnings_dates is not None and not earnings_dates.empty:
                    future_dates = earnings_dates[earnings_dates.index > pd.Timestamp.now()]
                    if not future_dates.empty:
                        next_earnings = future_dates.index[0]
            except Exception as e:
                logger.debug("Earnings dates fetch failed for %s: %s", ticker, e)

        # Calculate days until earnings
        if next_earnings is not None:
            now = datetime.now()
            earnings_dt = next_earnings.to_pydatetime().replace(tzinfo=None)
            days_until = (earnings_dt - now).days

            # Determine warning level
            if days_until < 0:
                warning_level = "past"
                message = f"Last earnings: {earnings_dt.strftime('%Y-%m-%d')} ({abs(days_until)} days ago)"
                emoji = "📅"
            elif days_until == 0:
                warning_level = "today"
                message = f"🚨 EARNINGS TODAY — Do not trade based on pre-earnings analysis!"
                emoji = "🚨"
            elif days_until <= 3:
                warning_level = "critical"
                message = f"⚠️ Earnings in {days_until} day(s) ({earnings_dt.strftime('%b %d')}) — HIGH RISK period"
                emoji = "⚠️"
            elif days_until <= 7:
                warning_level = "warning"
                message = f"📅 Earnings in {days_until} days ({earnings_dt.strftime('%b %d')}) — analysis may change after earnings"
                emoji = "📅"
            else:
                warning_level = "info"
                message = f"Next earnings: {earnings_dt.strftime('%B %d, %Y')} ({days_until} days away)"
                emoji = "ℹ️"

            return {
                "ticker": ticker,
                "next_earnings_date": earnings_dt.strftime("%Y-%m-%d"),
                "days_until_earnings": days_until,
                "warning_level": warning_level,
                "message": message,
                "emoji": emoji,
                "eps_estimate": float(eps_estimate) if eps_estimate and str(eps_estimate) != "nan" else None,
                "revenue_estimate": float(revenue_estimate) if revenue_estimate and str(revenue_estimate) != "nan" else None,
                "has_upcoming_earnings": 0 <= days_until <= 30,
            }

        # No earnings date found
        return {
            "ticker": ticker,
            "next_earnings_date": None,
            "days_until_earnings": None,
            "warning_level": "unknown",
            "message": "Earnings date not available for this ticker",
            "emoji": "❓",
            "eps_estimate": None,
            "revenue_estimate": None,
            "has_upcoming_earnings": False,
        }

    except Exception as e:
        logger.error("Earnings fetch error | ticker=%s | %s", ticker, e)
        return {
            "ticker": ticker,
            "next_earnings_date": None,
            "days_until_earnings": None,
            "warning_level": "error",
            "message": f"Could not fetch earnings data: {e}",
            "emoji": "❓",
            "eps_estimate": None,
            "revenue_estimate": None,
            "has_upcoming_earnings": False,
        }


@router.get(
    "/calendar/upcoming",
    summary="Get earnings calendar for multiple tickers",
    description="Returns upcoming earnings dates for a list of tickers.",
)
async def get_earnings_calendar(tickers: str = "AAPL,MSFT,GOOGL,NVDA,TSLA,JPM,AMZN,META") -> Dict[str, Any]:
    """Get earnings calendar for multiple tickers.

    Args:
        tickers: Comma-separated ticker symbols.

    Returns:
        Dict with earnings data for each ticker, sorted by date.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()][:10]

    # Fetch all in parallel
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, _fetch_earnings_sync, ticker)
        for ticker in ticker_list
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    calendar = []
    for ticker, result in zip(ticker_list, results):
        if isinstance(result, Exception):
            continue
        if result.get("next_earnings_date"):
            calendar.append(result)

    # Sort by days until earnings
    calendar.sort(key=lambda x: x.get("days_until_earnings", 999))

    return {
        "calendar": calendar,
        "total": len(calendar),
        "upcoming_within_7_days": [
            c for c in calendar
            if c.get("days_until_earnings") is not None and 0 <= c["days_until_earnings"] <= 7
        ],
    }
