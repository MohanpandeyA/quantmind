"""Live price chart WebSocket endpoint.

Streams real-time OHLCV candlestick data to the React frontend every 5 seconds.

WHY WEBSOCKET:
    HTTP polling: client asks "give me latest candle" every 5s
    → 12 requests/minute, each with HTTP handshake overhead (~50ms)
    → Candle data is delayed by up to 5 seconds

    WebSocket: server pushes new candle data as soon as it's available
    → 1 persistent connection, zero overhead after handshake
    → Same pattern as Bloomberg Terminal, TradingView, Robinhood

HOW IT WORKS:
    1. Client connects: ws://localhost:8000/live-chart/ws/AAPL
    2. Server immediately sends last 60 candles (history payload)
    3. Server loops every 5 seconds:
       - Fetches latest 1-minute OHLCV bar from yfinance
       - Pushes {type: "candle", open, high, low, close, volume, time}
    4. Client appends to rolling 60-candle buffer, re-renders chart
    5. On disconnect: server breaks loop, cleans up

DATA SHAPE:
    Initial history:
        {"type": "history", "candles": [...60 candles...], "ticker": "AAPL"}

    Live update (every 5s):
        {
            "type": "candle",
            "ticker": "AAPL",
            "time": "14:32",
            "timestamp": 1713700320,
            "open": 213.10,
            "high": 213.89,
            "low": 212.95,
            "close": 213.45,
            "volume": 1234567,
            "is_new_candle": true   ← false = update to current minute's candle
        }

    Error:
        {"type": "error", "message": "Invalid ticker: XYZ"}
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/live-chart", tags=["Live Chart"])

# How often to push a new candle update (seconds)
PUSH_INTERVAL_SECONDS = 5

# How many historical candles to send on connect
HISTORY_CANDLES = 60


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------

def _fetch_candles(ticker: str, period: str = "1d", interval: str = "1m") -> List[Dict[str, Any]]:
    """Fetch OHLCV candles from yfinance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').
        period: Data period ('1d', '5d', etc.).
        interval: Bar interval ('1m', '5m', '15m', etc.).

    Returns:
        List of candle dicts with keys: time, timestamp, open, high, low, close, volume.
        Empty list if fetch fails or market is closed.
    """
    try:
        import yfinance as yf

        df: pd.DataFrame = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

        if df.empty:
            logger.warning("live_chart | no data | ticker=%s", ticker)
            return []

        # Flatten MultiIndex columns if present (yfinance ≥0.2.40)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        candles = []
        for ts, row in df.iterrows():
            # ts is a pandas Timestamp
            try:
                open_  = float(row["Open"])
                high   = float(row["High"])
                low    = float(row["Low"])
                close  = float(row["Close"])
                volume = int(row["Volume"])
            except (KeyError, ValueError, TypeError):
                continue

            # Skip rows with NaN
            if any(math.isnan(v) for v in [open_, high, low, close]):
                continue

            # Format time label for x-axis
            if hasattr(ts, "strftime"):
                time_label = ts.strftime("%H:%M")
                unix_ts = int(ts.timestamp())
            else:
                time_label = str(ts)
                unix_ts = 0

            candles.append({
                "time":      time_label,
                "timestamp": unix_ts,
                "open":      round(open_,  4),
                "high":      round(high,   4),
                "low":       round(low,    4),
                "close":     round(close,  4),
                "volume":    volume,
            })

        return candles

    except Exception as e:
        logger.error("live_chart | fetch failed | ticker=%s | %s", ticker, e)
        return []


def _sanitize(candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace inf/nan with 0 for JSON serialization safety."""
    clean = []
    for c in candles:
        clean.append({
            k: (0 if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
            for k, v in c.items()
        })
    return clean


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/{ticker}")
async def live_chart_ws(websocket: WebSocket, ticker: str) -> None:
    """Stream real-time OHLCV candlestick data for a ticker.

    Protocol:
        1. Accept connection
        2. Validate ticker (basic check)
        3. Send history payload (last 60 candles)
        4. Loop every PUSH_INTERVAL_SECONDS:
           - Fetch latest candle
           - Push update
        5. On disconnect: break loop

    Args:
        websocket: FastAPI WebSocket connection.
        ticker: Stock ticker symbol from URL path (e.g., 'AAPL').
    """
    ticker = ticker.upper().strip()
    await websocket.accept()

    logger.info("live_chart | connected | ticker=%s", ticker)

    # Basic ticker validation
    if not ticker or len(ticker) > 12 or not ticker.replace(".", "").replace("-", "").isalnum():
        await websocket.send_json({
            "type": "error",
            "message": f"Invalid ticker: {ticker}",
        })
        await websocket.close()
        return

    try:
        # --- Step 1: Send historical candles immediately on connect ---
        loop = asyncio.get_event_loop()
        history = await loop.run_in_executor(
            None, lambda: _fetch_candles(ticker, period="1d", interval="1m")
        )
        history = _sanitize(history)

        # Keep last HISTORY_CANDLES candles
        history = history[-HISTORY_CANDLES:]

        await websocket.send_json({
            "type":    "history",
            "ticker":  ticker,
            "candles": history,
            "count":   len(history),
        })

        logger.info(
            "live_chart | history sent | ticker=%s | candles=%d",
            ticker, len(history),
        )

        # Track the last candle timestamp to detect new candles
        last_timestamp: Optional[int] = history[-1]["timestamp"] if history else None

        # --- Step 2: Stream live updates every PUSH_INTERVAL_SECONDS ---
        while True:
            await asyncio.sleep(PUSH_INTERVAL_SECONDS)

            # Fetch latest candles (last 5 minutes to catch the current candle)
            latest = await loop.run_in_executor(
                None, lambda: _fetch_candles(ticker, period="1d", interval="1m")
            )
            latest = _sanitize(latest)

            if not latest:
                # Market closed or fetch failed — send heartbeat
                await websocket.send_json({
                    "type":    "heartbeat",
                    "ticker":  ticker,
                    "message": "Market may be closed or data unavailable",
                    "time":    datetime.now(timezone.utc).strftime("%H:%M:%S"),
                })
                continue

            # Get the most recent candle
            newest = latest[-1]
            is_new_candle = (newest["timestamp"] != last_timestamp)

            if is_new_candle:
                last_timestamp = newest["timestamp"]

            await websocket.send_json({
                "type":          "candle",
                "ticker":        ticker,
                "time":          newest["time"],
                "timestamp":     newest["timestamp"],
                "open":          newest["open"],
                "high":          newest["high"],
                "low":           newest["low"],
                "close":         newest["close"],
                "volume":        newest["volume"],
                "is_new_candle": is_new_candle,
                "server_time":   datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            })

            logger.debug(
                "live_chart | pushed | ticker=%s | close=%.2f | new=%s",
                ticker, newest["close"], is_new_candle,
            )

    except WebSocketDisconnect:
        logger.info("live_chart | disconnected | ticker=%s", ticker)
    except Exception as e:
        logger.error("live_chart | error | ticker=%s | %s", ticker, e, exc_info=True)
        try:
            await websocket.send_json({
                "type":    "error",
                "message": f"Stream error: {e}",
            })
        except Exception:
            pass
