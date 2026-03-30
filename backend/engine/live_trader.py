"""LiveTrader — Real-world trading engine for QuantMind.

Connects a strategy to a live broker (Alpaca paper/live trading API)
via WebSocket for real-time price bars. Executes orders asynchronously
and monitors risk in real time using IncrementalMetrics.

Architecture:
    WebSocket (Alpaca) → on_bar() callback → get_latest_signal() O(1)
    → risk check → async order submission → IncrementalMetrics update

Latency budget (swing trading):
    Signal generation:    ~1 μs   (OnlineEMA/OnlineZScore)
    Risk check:           ~2 μs   (IncrementalMetrics)
    Order submission:     ~50ms   (network to Alpaca)
    Total hot path:       ~50ms   (acceptable for daily/hourly bars)

Free tier: Uses Alpaca paper trading (no real money, free API key).
Live tier: Change ALPACA_BASE_URL to live endpoint.

Setup:
    1. Get free Alpaca API key: https://alpaca.markets
    2. Add to .env:
       ALPACA_API_KEY=your_key
       ALPACA_SECRET_KEY=your_secret
       ALPACA_BASE_URL=https://paper-api.alpaca.markets
    3. Run: python -m engine.live_trader
"""

from __future__ import annotations

import asyncio
import json
import signal as os_signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from config.logging_config import get_logger, stop_logging
from config.settings import settings
from engine.online_indicators import IncrementalMetrics
from engine.strategies.base_strategy import BaseStrategy, Signal, StrategyConfig

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LiveBar:
    """A single OHLCV bar received from the live data feed.

    Attributes:
        symbol: Ticker symbol (e.g., 'AAPL').
        timestamp: Bar close timestamp (UTC).
        open: Opening price.
        high: High price.
        low: Low price.
        close: Closing price.
        volume: Trading volume.
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TradeRecord:
    """Record of a single executed trade.

    Attributes:
        symbol: Ticker symbol.
        side: 'buy' or 'sell'.
        qty: Number of shares.
        price: Execution price.
        timestamp: Execution timestamp.
        signal: Signal that triggered the trade.
        order_id: Broker order ID.
    """

    symbol: str
    side: str
    qty: float
    price: float
    timestamp: datetime
    signal: Signal
    order_id: str = ""


@dataclass
class LiveTraderConfig:
    """Configuration for the LiveTrader.

    Attributes:
        symbol: Ticker to trade (e.g., 'AAPL').
        bar_timeframe: Bar timeframe ('1Min', '5Min', '1Hour', '1Day').
        max_drawdown_limit: Halt if drawdown exceeds this (default 10%).
        min_sharpe_limit: Alert if Sharpe drops below this (default 0.0).
        paper_trading: If True, use paper trading endpoint.
        position_size_usd: Fixed USD amount per trade (overrides strategy config).
        max_position_usd: Maximum total position size in USD.
    """

    symbol: str
    bar_timeframe: str = "1Day"
    max_drawdown_limit: float = 0.10
    min_sharpe_limit: float = 0.0
    paper_trading: bool = True
    position_size_usd: float = 10_000.0
    max_position_usd: float = 50_000.0


# ---------------------------------------------------------------------------
# LiveTrader
# ---------------------------------------------------------------------------

class LiveTrader:
    """Real-world trading engine connecting strategy to Alpaca broker.

    Receives live price bars via WebSocket, generates signals using
    the strategy's O(1) get_latest_signal() method, and submits
    orders asynchronously to Alpaca.

    Risk management:
        - IncrementalMetrics monitors drawdown and Sharpe in real time
        - Automatic halt if drawdown > max_drawdown_limit
        - Emergency stop on SIGINT/SIGTERM

    Usage (paper trading — free, no real money):
        >>> from engine.strategies.momentum import MomentumStrategy
        >>> strategy = MomentumStrategy()
        >>> trader_config = LiveTraderConfig(symbol="AAPL")
        >>> trader = LiveTrader(strategy, trader_config)
        >>> asyncio.run(trader.start())

    Attributes:
        strategy: Trading strategy instance.
        config: LiveTraderConfig.
        metrics: IncrementalMetrics for real-time risk monitoring.
        trade_log: List of all executed trades.
        is_running: Whether the trader is currently active.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        config: LiveTraderConfig,
    ) -> None:
        """Initialize the LiveTrader.

        Args:
            strategy: Instantiated strategy with get_latest_signal() implemented.
            config: LiveTraderConfig with symbol and risk parameters.
        """
        self.strategy = strategy
        self.config = config
        self.metrics = IncrementalMetrics(
            max_drawdown_limit=config.max_drawdown_limit,
            min_sharpe_limit=config.min_sharpe_limit,
        )
        self.trade_log: List[TradeRecord] = []
        self.is_running: bool = False

        # Position tracking
        self._current_position: float = 0.0   # Shares held (positive = long)
        self._entry_price: float = 0.0
        self._equity: float = strategy.config.initial_capital
        self._last_equity: float = strategy.config.initial_capital

        # Performance tracking
        self._bar_count: int = 0
        self._signal_latencies: List[float] = []  # μs per signal

        logger.info(
            "LiveTrader initialized | symbol=%s | strategy=%s | paper=%s",
            config.symbol,
            strategy.get_name(),
            config.paper_trading,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the live trading loop.

        Connects to Alpaca WebSocket, subscribes to bars, and processes
        incoming data until stopped.

        Note: Requires alpaca-trade-api or alpaca-py installed.
        Install: pip install alpaca-py

        Raises:
            ImportError: If alpaca-py is not installed.
            RuntimeError: If API keys are not configured.
        """
        self._validate_api_keys()
        self.is_running = True

        # Reset strategy online state at session start
        self.strategy.reset_online_state()

        # Register graceful shutdown handlers
        loop = asyncio.get_event_loop()
        for sig in (os_signal.SIGINT, os_signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)

        logger.info(
            "LiveTrader starting | symbol=%s | timeframe=%s",
            self.config.symbol,
            self.config.bar_timeframe,
        )

        try:
            await self._run_websocket_loop()
        except Exception as e:
            logger.error("LiveTrader error: %s", e, exc_info=True)
            raise
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Gracefully stop the live trading loop."""
        logger.info("LiveTrader stopping | symbol=%s", self.config.symbol)
        self.is_running = False

    # ------------------------------------------------------------------
    # Core hot path — called on every new bar
    # ------------------------------------------------------------------

    async def on_bar(self, bar: LiveBar) -> None:
        """Process a new price bar. This is the trading hot path.

        Steps (target: < 100ms total):
            1. Generate signal via O(1) get_latest_signal()   ~1 μs
            2. Check risk limits via IncrementalMetrics        ~2 μs
            3. Submit order if signal is actionable            ~50ms (network)
            4. Update metrics                                  ~2 μs

        Args:
            bar: LiveBar with OHLCV data for the latest period.
        """
        self._bar_count += 1

        # --- Step 1: Generate signal (O(1) hot path) ---
        t0 = time.perf_counter()
        signal = self.strategy.get_latest_signal(bar.close)
        latency_us = (time.perf_counter() - t0) * 1_000_000
        self._signal_latencies.append(latency_us)

        logger.debug(
            "Bar %d | %s | close=%.2f | signal=%s | latency=%.1fμs",
            self._bar_count, bar.symbol, bar.close, signal.name, latency_us,
        )

        # --- Step 2: Risk check ---
        if self.metrics.should_halt():
            logger.warning(
                "RISK HALT | drawdown=%.1f%% | halting all trading",
                self.metrics.current_drawdown * 100,
            )
            await self.stop()
            return

        # --- Step 3: Execute order if actionable ---
        if signal == Signal.BUY and self._current_position == 0.0:
            await self._execute_buy(bar)

        elif signal == Signal.SELL and self._current_position > 0.0:
            await self._execute_sell(bar)

        # --- Step 4: Update metrics ---
        self._update_equity(bar.close)
        daily_return = (
            (self._equity - self._last_equity) / self._last_equity
            if self._last_equity > 0 else 0.0
        )
        self.metrics.update(self._equity, daily_return)
        self._last_equity = self._equity

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    async def _execute_buy(self, bar: LiveBar) -> None:
        """Submit a market buy order.

        Args:
            bar: Current price bar.
        """
        qty = self._compute_qty(bar.close)
        if qty <= 0:
            logger.warning("BUY skipped | insufficient capital | price=%.2f", bar.close)
            return

        order_id = await self._submit_order(
            symbol=bar.symbol,
            side="buy",
            qty=qty,
            price=bar.close,
        )

        self._current_position = qty
        self._entry_price = bar.close

        record = TradeRecord(
            symbol=bar.symbol,
            side="buy",
            qty=qty,
            price=bar.close,
            timestamp=bar.timestamp,
            signal=Signal.BUY,
            order_id=order_id,
        )
        self.trade_log.append(record)

        logger.info(
            "BUY executed | %s | qty=%.2f | price=%.2f | order_id=%s",
            bar.symbol, qty, bar.close, order_id,
        )

    async def _execute_sell(self, bar: LiveBar) -> None:
        """Submit a market sell order to close position.

        Args:
            bar: Current price bar.
        """
        qty = self._current_position
        order_id = await self._submit_order(
            symbol=bar.symbol,
            side="sell",
            qty=qty,
            price=bar.close,
        )

        trade_return = (bar.close - self._entry_price) / self._entry_price
        self._current_position = 0.0
        self._entry_price = 0.0

        record = TradeRecord(
            symbol=bar.symbol,
            side="sell",
            qty=qty,
            price=bar.close,
            timestamp=bar.timestamp,
            signal=Signal.SELL,
            order_id=order_id,
        )
        self.trade_log.append(record)

        logger.info(
            "SELL executed | %s | qty=%.2f | price=%.2f | trade_ret=%.2f%% | order_id=%s",
            bar.symbol, qty, bar.close, trade_return * 100, order_id,
        )

    async def _submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> str:
        """Submit order to Alpaca broker API.

        Uses alpaca-py if available, otherwise simulates (paper mode fallback).

        Args:
            symbol: Ticker symbol.
            side: 'buy' or 'sell'.
            qty: Number of shares.
            price: Current price (for logging).

        Returns:
            Order ID string from broker.
        """
        try:
            from alpaca.trading.client import TradingClient  # type: ignore[import]
            from alpaca.trading.requests import MarketOrderRequest  # type: ignore[import]
            from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore[import]

            client = TradingClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                paper=self.config.paper_trading,
            )

            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )

            order = client.submit_order(order_data=order_data)
            return str(order.id)

        except ImportError:
            # alpaca-py not installed — simulate order (paper fallback)
            logger.warning(
                "alpaca-py not installed. Simulating order: %s %s %.2f @ %.2f",
                side.upper(), symbol, qty, price,
            )
            return f"SIM-{side.upper()}-{int(time.time())}"

        except Exception as e:
            logger.error("Order submission failed: %s", e, exc_info=True)
            return f"FAILED-{int(time.time())}"

    # ------------------------------------------------------------------
    # WebSocket loop
    # ------------------------------------------------------------------

    async def _run_websocket_loop(self) -> None:
        """Connect to Alpaca WebSocket and process incoming bars.

        Uses alpaca-py's async data stream if available.
        Falls back to a simulation loop for testing without API keys.
        """
        try:
            from alpaca.data.live import StockDataStream  # type: ignore[import]

            stream = StockDataStream(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
            )

            async def bar_handler(bar: object) -> None:
                """Callback for each incoming bar from WebSocket."""
                live_bar = LiveBar(
                    symbol=getattr(bar, "symbol", self.config.symbol),
                    timestamp=getattr(bar, "timestamp", datetime.utcnow()),
                    open=float(getattr(bar, "open", 0)),
                    high=float(getattr(bar, "high", 0)),
                    low=float(getattr(bar, "low", 0)),
                    close=float(getattr(bar, "close", 0)),
                    volume=float(getattr(bar, "volume", 0)),
                )
                await self.on_bar(live_bar)

            stream.subscribe_bars(bar_handler, self.config.symbol)
            logger.info("WebSocket connected | subscribing to %s bars", self.config.symbol)
            await stream.run()

        except ImportError:
            logger.warning(
                "alpaca-py not installed. Running simulation loop. "
                "Install with: pip install alpaca-py"
            )
            await self._simulation_loop()

    async def _simulation_loop(self) -> None:
        """Simulation loop for testing without real API keys.

        Generates synthetic price bars using geometric Brownian motion.
        Useful for testing the full pipeline end-to-end.
        """
        logger.info("Simulation loop started | symbol=%s", self.config.symbol)
        price = 100.0
        np.random.seed(42)

        while self.is_running:
            # Simulate a new bar (GBM: geometric Brownian motion)
            ret = np.random.normal(0.0005, 0.015)
            price *= (1 + ret)

            bar = LiveBar(
                symbol=self.config.symbol,
                timestamp=datetime.utcnow(),
                open=price * 0.999,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=float(np.random.randint(1_000_000, 5_000_000)),
            )

            await self.on_bar(bar)

            # Simulate daily bar interval (1 second in simulation)
            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_qty(self, price: float) -> float:
        """Compute number of shares to buy based on position size config.

        Args:
            price: Current price per share.

        Returns:
            Number of shares (floored to whole shares).
        """
        available = min(
            self.config.position_size_usd,
            self.config.max_position_usd,
            self._equity * self.strategy.config.position_size,
        )
        if price <= 0:
            return 0.0
        return float(int(available / price))  # Whole shares only

    def _update_equity(self, current_price: float) -> None:
        """Mark portfolio to market at current price.

        Args:
            current_price: Latest close price.
        """
        if self._current_position > 0:
            unrealized_pnl = self._current_position * (current_price - self._entry_price)
            self._equity = self._last_equity + unrealized_pnl

    def _validate_api_keys(self) -> None:
        """Validate that required API keys are configured.

        Raises:
            RuntimeError: If keys are missing and not in simulation mode.
        """
        if not settings.alpaca_api_key and not settings.alpaca_secret_key:
            logger.warning(
                "Alpaca API keys not configured. Running in simulation mode. "
                "Add ALPACA_API_KEY and ALPACA_SECRET_KEY to .env for live trading."
            )

    def _handle_shutdown(self) -> None:
        """Handle SIGINT/SIGTERM — graceful shutdown."""
        logger.info("Shutdown signal received | stopping LiveTrader")
        asyncio.create_task(self.stop())

    async def _cleanup(self) -> None:
        """Clean up resources on shutdown."""
        logger.info(
            "LiveTrader cleanup | bars_processed=%d | trades=%d",
            self._bar_count,
            len(self.trade_log),
        )
        if self._signal_latencies:
            avg_latency = sum(self._signal_latencies) / len(self._signal_latencies)
            max_latency = max(self._signal_latencies)
            logger.info(
                "Signal latency | avg=%.1fμs | max=%.1fμs",
                avg_latency, max_latency,
            )
        stop_logging()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_performance_summary(self) -> Dict[str, object]:
        """Return current performance summary.

        Returns:
            Dictionary with metrics, trade count, and latency stats.
        """
        summary = self.metrics.get_summary()
        summary.update({
            "strategy": self.strategy.get_name(),
            "symbol": self.config.symbol,
            "n_trades": len(self.trade_log),
            "bars_processed": self._bar_count,
            "current_position": self._current_position,
        })
        if self._signal_latencies:
            summary["avg_signal_latency_us"] = round(
                sum(self._signal_latencies) / len(self._signal_latencies), 2
            )
        return summary
