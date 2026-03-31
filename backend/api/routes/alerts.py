"""Real-time price alerts via WebSocket.

WHY WEBSOCKET INSTEAD OF HTTP POLLING:
    HTTP polling: client asks "any alerts?" every 5 seconds
    - 12 requests/minute × 60 minutes = 720 requests/hour
    - Each request has HTTP overhead (~50ms)
    - Alerts are delayed by up to 5 seconds

    WebSocket: persistent connection, server pushes instantly
    - 1 connection maintained
    - Zero overhead after handshake
    - Alerts delivered in <1 second of threshold crossing
    - This is how Bloomberg terminals, trading platforms work

HOW IT WORKS:
    1. Client connects: ws://localhost:8000/alerts/ws
    2. Client sends: {"action": "add", "ticker": "AAPL", "condition": "below", "threshold": 200.0}
    3. Server polls yfinance every 30 seconds for all watched tickers
    4. When AAPL drops below $200: server pushes {"ticker": "AAPL", "alert": "PRICE BELOW $200", ...}
    5. Client shows notification

ALERT CONDITIONS:
    - price_below: alert when price drops below threshold (buy opportunity)
    - price_above: alert when price rises above threshold (take profit / stop loss)
    - change_pct_below: alert when daily % change drops below threshold (e.g., -5%)
    - change_pct_above: alert when daily % change rises above threshold (e.g., +5%)

DESIGN PATTERN — Connection Manager:
    Multiple clients can connect simultaneously. The ConnectionManager
    tracks all active WebSocket connections and broadcasts to all of them.
    This is the standard pattern for WebSocket servers.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/alerts", tags=["Alerts"])

# ---------------------------------------------------------------------------
# Alert data structures
# ---------------------------------------------------------------------------

@dataclass
class PriceAlert:
    """A single price alert configuration.

    Attributes:
        ticker: Stock ticker to watch.
        condition: Alert condition type.
        threshold: Threshold value to trigger alert.
        message: Custom message to show when triggered.
        triggered: Whether this alert has already fired.
    """

    ticker: str
    condition: str  # "price_below", "price_above", "change_pct_below", "change_pct_above"
    threshold: float
    message: str = ""
    triggered: bool = False

    def check(self, current_price: float, change_pct: float) -> bool:
        """Check if this alert should trigger.

        Args:
            current_price: Current stock price.
            change_pct: Today's % change.

        Returns:
            True if alert condition is met and not already triggered.
        """
        if self.triggered:
            return False

        if self.condition == "price_below" and current_price < self.threshold:
            return True
        if self.condition == "price_above" and current_price > self.threshold:
            return True
        if self.condition == "change_pct_below" and change_pct < self.threshold:
            return True
        if self.condition == "change_pct_above" and change_pct > self.threshold:
            return True
        return False

    def format_message(self, current_price: float, change_pct: float) -> str:
        """Format the alert message.

        Args:
            current_price: Current price.
            change_pct: Today's % change.

        Returns:
            Formatted alert message string.
        """
        condition_text = {
            "price_below": f"dropped below ${self.threshold:.2f}",
            "price_above": f"rose above ${self.threshold:.2f}",
            "change_pct_below": f"fell {self.threshold:.1f}% today",
            "change_pct_above": f"gained {self.threshold:.1f}% today",
        }.get(self.condition, f"crossed {self.threshold}")

        custom = f" — {self.message}" if self.message else ""
        return (
            f"🚨 {self.ticker} {condition_text} "
            f"(current: ${current_price:.2f}, {change_pct:+.2f}% today){custom}"
        )


# ---------------------------------------------------------------------------
# Connection Manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages all active WebSocket connections.

    WHY THIS CLASS:
        Multiple browser tabs / clients can connect simultaneously.
        This manager tracks all connections and broadcasts alerts to all of them.
        When a client disconnects, it's cleanly removed from the set.

    Attributes:
        active_connections: Set of active WebSocket connections.
        alerts: List of configured price alerts.
        _monitor_task: Background task that polls prices.
    """

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self.alerts: List[PriceAlert] = []
        self._monitor_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection.

        Args:
            websocket: New WebSocket connection.
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(
            "WebSocket connected | total_connections=%d",
            len(self.active_connections),
        )

        # Start monitoring if not already running
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_prices())

        # Send current alerts to new connection
        await websocket.send_json({
            "type": "connected",
            "message": f"Connected to QuantMind alerts. {len(self.alerts)} alerts active.",
            "alerts": [
                {"ticker": a.ticker, "condition": a.condition, "threshold": a.threshold}
                for a in self.alerts
            ],
        })

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket.

        Args:
            websocket: Disconnected WebSocket.
        """
        self.active_connections.discard(websocket)
        logger.info(
            "WebSocket disconnected | total_connections=%d",
            len(self.active_connections),
        )

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Send a message to all connected clients.

        Args:
            message: JSON-serializable message dict.
        """
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections.discard(conn)

    def add_alert(self, alert: PriceAlert) -> None:
        """Add a new price alert.

        Args:
            alert: PriceAlert to add.
        """
        # Remove existing alert for same ticker+condition
        self.alerts = [
            a for a in self.alerts
            if not (a.ticker == alert.ticker and a.condition == alert.condition)
        ]
        self.alerts.append(alert)
        logger.info(
            "Alert added | ticker=%s | condition=%s | threshold=%.2f",
            alert.ticker, alert.condition, alert.threshold,
        )

    def remove_alert(self, ticker: str, condition: Optional[str] = None) -> int:
        """Remove alerts for a ticker.

        Args:
            ticker: Ticker to remove alerts for.
            condition: Optional specific condition to remove.

        Returns:
            Number of alerts removed.
        """
        before = len(self.alerts)
        if condition:
            self.alerts = [a for a in self.alerts if not (a.ticker == ticker and a.condition == condition)]
        else:
            self.alerts = [a for a in self.alerts if a.ticker != ticker]
        removed = before - len(self.alerts)
        logger.info("Alerts removed | ticker=%s | count=%d", ticker, removed)
        return removed

    async def _monitor_prices(self) -> None:
        """Background task: poll prices every 30s and check alert conditions.

        WHY 30 SECONDS:
            - Yahoo Finance rate limit: 10 req/sec
            - 30s interval is respectful and sufficient for swing trading alerts
            - For HFT (millisecond alerts), you'd need a paid real-time feed
        """
        logger.info("Price monitor started | polling every 30s")

        while self.active_connections:
            try:
                await self._check_all_alerts()
            except Exception as e:
                logger.error("Monitor error: %s", e)

            await asyncio.sleep(30)

        logger.info("Price monitor stopped | no active connections")

    async def _check_all_alerts(self) -> None:
        """Fetch current prices and check all alert conditions."""
        if not self.alerts:
            return

        # Get unique tickers
        tickers = list({a.ticker for a in self.alerts if not a.triggered})
        if not tickers:
            return

        # Fetch prices
        prices = await _fetch_prices_batch(tickers)

        # Check each alert
        triggered_alerts = []
        for alert in self.alerts:
            if alert.triggered:
                continue

            price_data = prices.get(alert.ticker, {})
            if not price_data:
                continue

            current_price = price_data.get("current_price", 0)
            change_pct = price_data.get("price_change_pct", 0)

            if alert.check(current_price, change_pct):
                alert.triggered = True
                triggered_alerts.append({
                    "type": "alert",
                    "ticker": alert.ticker,
                    "condition": alert.condition,
                    "threshold": alert.threshold,
                    "current_price": current_price,
                    "change_pct": change_pct,
                    "message": alert.format_message(current_price, change_pct),
                    "timestamp": datetime.utcnow().isoformat(),
                })

        # Broadcast triggered alerts
        for alert_msg in triggered_alerts:
            logger.info("Alert triggered | %s", alert_msg["message"])
            await self.broadcast(alert_msg)

        # Send price update to all clients
        if prices:
            await self.broadcast({
                "type": "price_update",
                "prices": prices,
                "timestamp": datetime.utcnow().isoformat(),
            })


# Singleton connection manager
manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Helper: batch price fetch
# ---------------------------------------------------------------------------

async def _fetch_prices_batch(tickers: List[str]) -> Dict[str, Dict[str, float]]:
    """Fetch current prices for multiple tickers.

    Args:
        tickers: List of ticker symbols.

    Returns:
        Dict mapping ticker → price data.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_sync, tickers)


def _fetch_sync(tickers: List[str]) -> Dict[str, Dict[str, float]]:
    """Synchronous price fetch.

    Args:
        tickers: List of tickers.

    Returns:
        Price data dict.
    """
    warnings.filterwarnings("ignore")
    try:
        import yfinance as yf
        data = yf.download(tickers, period="1d", auto_adjust=True, progress=False)
        result: Dict[str, Dict[str, float]] = {}

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    hist = data
                else:
                    hist = data[ticker] if ticker in data.columns.get_level_values(0) else data

                if hist.empty:
                    continue

                if hasattr(hist.columns, "levels"):
                    hist.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in hist.columns]
                else:
                    hist.columns = [c.lower() for c in hist.columns]

                current = float(hist["close"].iloc[-1])
                prev = float(hist["open"].iloc[-1])
                change_pct = ((current - prev) / prev * 100) if prev > 0 else 0.0

                result[ticker] = {
                    "current_price": round(current, 2),
                    "price_change_pct": round(change_pct, 2),
                }
            except Exception:
                pass

        return result
    except Exception as e:
        logger.error("Price fetch error: %s", e)
        return {}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws")
async def websocket_alerts(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time price alerts.

    Client messages (JSON):
        Add alert:    {"action": "add", "ticker": "AAPL", "condition": "price_below", "threshold": 200.0, "message": "Buy opportunity"}
        Remove alert: {"action": "remove", "ticker": "AAPL"}
        List alerts:  {"action": "list"}
        Ping:         {"action": "ping"}

    Server messages (JSON):
        Connected:      {"type": "connected", "message": "...", "alerts": [...]}
        Alert fired:    {"type": "alert", "ticker": "AAPL", "message": "🚨 AAPL dropped below $200", ...}
        Price update:   {"type": "price_update", "prices": {...}, "timestamp": "..."}
        Alert added:    {"type": "alert_added", "ticker": "AAPL", ...}
        Pong:           {"type": "pong"}
    """
    await manager.connect(websocket)

    try:
        while True:
            # Wait for client message
            data = await websocket.receive_text()

            try:
                msg = json.loads(data)
                action = msg.get("action", "")

                if action == "add":
                    alert = PriceAlert(
                        ticker=msg["ticker"].upper(),
                        condition=msg.get("condition", "price_below"),
                        threshold=float(msg.get("threshold", 0)),
                        message=msg.get("message", ""),
                    )
                    manager.add_alert(alert)
                    await websocket.send_json({
                        "type": "alert_added",
                        "ticker": alert.ticker,
                        "condition": alert.condition,
                        "threshold": alert.threshold,
                        "message": f"Alert set: {alert.ticker} {alert.condition} {alert.threshold}",
                    })

                elif action == "remove":
                    ticker = msg.get("ticker", "").upper()
                    removed = manager.remove_alert(ticker, msg.get("condition"))
                    await websocket.send_json({
                        "type": "alert_removed",
                        "ticker": ticker,
                        "count": removed,
                    })

                elif action == "list":
                    await websocket.send_json({
                        "type": "alert_list",
                        "alerts": [
                            {
                                "ticker": a.ticker,
                                "condition": a.condition,
                                "threshold": a.threshold,
                                "triggered": a.triggered,
                            }
                            for a in manager.alerts
                        ],
                    })

                elif action == "ping":
                    await websocket.send_json({"type": "pong"})

            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            except KeyError as e:
                await websocket.send_json({"type": "error", "message": f"Missing field: {e}"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
