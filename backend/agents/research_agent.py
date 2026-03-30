"""ResearchAgent — fetches real-time market data for a ticker.

This is the FIRST agent in the LangGraph workflow. It fetches:
- Current price, volume, market cap, P/E ratio
- 52-week high/low
- Recent OHLCV price history for charting

Data source: yfinance (Yahoo Finance) — completely free, no API key.

Why this is the first agent:
    All subsequent agents need market context. The StrategyAgent needs
    to know if the stock is trending or mean-reverting. The ExplainerAgent
    needs current price to contextualize the recommendation.

LangGraph node contract:
    Input:  TradingState with ticker set
    Output: TradingState with market_data and price_history populated
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from config.logging_config import get_logger
from graph.state import MarketData, TradingState

logger = get_logger(__name__)


async def research_agent(state: TradingState) -> TradingState:
    """Fetch market data for the ticker. LangGraph node function.

    Fetches current price, fundamentals, and recent price history
    using yfinance (free, no API key required).

    Args:
        state: Current TradingState with ticker set.

    Returns:
        Updated TradingState with market_data and price_history populated.
        On failure, sets state["error"] and returns with empty market data.

    Example:
        >>> state = create_initial_state("AAPL", "Should I buy AAPL?")
        >>> state = await research_agent(state)
        >>> state["market_data"]["current_price"]
        189.50
    """
    ticker = state.get("ticker", "")
    logger.info("ResearchAgent | fetching | ticker=%s", ticker)

    try:
        import yfinance as yf  # type: ignore[import]

        stock = yf.Ticker(ticker)

        # Fetch current info
        info = stock.info or {}

        # Fetch recent price history (1 year for charting)
        hist = stock.history(period="1y", auto_adjust=True)

        if hist.empty:
            logger.warning("ResearchAgent | no price data | ticker=%s", ticker)
            return {**state, "error": f"No price data found for ticker '{ticker}'."}

        # Build market data
        current_price = float(hist["Close"].iloc[-1])
        prev_price = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
        price_change_pct = ((current_price - prev_price) / prev_price) * 100

        market_data = MarketData(
            ticker=ticker,
            current_price=round(current_price, 2),
            price_change_pct=round(price_change_pct, 2),
            volume=float(hist["Volume"].iloc[-1]),
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE"),
            week_52_high=float(hist["High"].max()),
            week_52_low=float(hist["Low"].min()),
            avg_volume=float(hist["Volume"].mean()),
        )

        # Build price history for charting (last 252 trading days)
        price_history: List[Dict[str, Any]] = []
        for date, row in hist.tail(252).iterrows():
            price_history.append({
                "date": str(date.date()),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })

        logger.info(
            "ResearchAgent | complete | ticker=%s | price=%.2f | change=%.2f%%",
            ticker, current_price, price_change_pct,
        )

        return {
            **state,
            "market_data": market_data,
            "price_history": price_history,
        }

    except Exception as e:
        logger.error("ResearchAgent | failed | ticker=%s | %s", ticker, e, exc_info=True)
        return {**state, "error": f"ResearchAgent failed: {e}"}
