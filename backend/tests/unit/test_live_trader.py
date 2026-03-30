"""Unit tests for LiveTrader simulation mode.

Tests cover:
- LiveTraderConfig validation
- LiveTrader initialization
- on_bar() signal processing and order execution
- Risk halt on drawdown breach
- Trade log recording
- Performance summary
- Simulation loop (no API keys needed)
- Reset and cleanup
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

from engine.live_trader import LiveBar, LiveTrader, LiveTraderConfig, TradeRecord
from engine.strategies.base_strategy import Signal, StrategyConfig
from engine.strategies.momentum import MomentumStrategy
from engine.strategies.mean_reversion import MeanReversionStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def momentum_strategy() -> MomentumStrategy:
    return MomentumStrategy(
        StrategyConfig(
            initial_capital=100_000.0,
            position_size=0.5,
            params={"short_window": 5, "long_window": 10},
        )
    )


@pytest.fixture
def trader_config() -> LiveTraderConfig:
    return LiveTraderConfig(
        symbol="AAPL",
        bar_timeframe="1Day",
        max_drawdown_limit=0.15,
        paper_trading=True,
        position_size_usd=10_000.0,
    )


@pytest.fixture
def live_trader(
    momentum_strategy: MomentumStrategy,
    trader_config: LiveTraderConfig,
) -> LiveTrader:
    return LiveTrader(momentum_strategy, trader_config)


def make_bar(
    symbol: str = "AAPL",
    close: float = 150.0,
    open_: float = 149.0,
    high: float = 151.0,
    low: float = 148.0,
    volume: float = 1_000_000.0,
) -> LiveBar:
    return LiveBar(
        symbol=symbol,
        timestamp=datetime.utcnow(),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


# ---------------------------------------------------------------------------
# LiveTraderConfig tests
# ---------------------------------------------------------------------------

class TestLiveTraderConfig:
    """Tests for LiveTraderConfig dataclass."""

    def test_default_values(self) -> None:
        config = LiveTraderConfig(symbol="AAPL")
        assert config.symbol == "AAPL"
        assert config.bar_timeframe == "1Day"
        assert config.max_drawdown_limit == 0.10
        assert config.paper_trading is True

    def test_custom_values(self) -> None:
        config = LiveTraderConfig(
            symbol="MSFT",
            bar_timeframe="1Hour",
            max_drawdown_limit=0.05,
            paper_trading=False,
            position_size_usd=5_000.0,
        )
        assert config.symbol == "MSFT"
        assert config.max_drawdown_limit == 0.05
        assert config.paper_trading is False


# ---------------------------------------------------------------------------
# LiveTrader initialization tests
# ---------------------------------------------------------------------------

class TestLiveTraderInit:
    """Tests for LiveTrader initialization."""

    def test_initializes_correctly(
        self,
        live_trader: LiveTrader,
        momentum_strategy: MomentumStrategy,
        trader_config: LiveTraderConfig,
    ) -> None:
        assert live_trader.strategy is momentum_strategy
        assert live_trader.config is trader_config
        assert live_trader.is_running is False
        assert live_trader._current_position == 0.0
        assert len(live_trader.trade_log) == 0

    def test_metrics_initialized(self, live_trader: LiveTrader) -> None:
        assert live_trader.metrics is not None
        assert live_trader.metrics.current_drawdown == 0.0

    def test_equity_initialized_to_capital(
        self, live_trader: LiveTrader, momentum_strategy: MomentumStrategy
    ) -> None:
        assert live_trader._equity == momentum_strategy.config.initial_capital


# ---------------------------------------------------------------------------
# on_bar() processing tests
# ---------------------------------------------------------------------------

class TestOnBarProcessing:
    """Tests for the on_bar() hot path."""

    def test_hold_signal_no_trade(self, live_trader: LiveTrader) -> None:
        """HOLD signal should not create any trades."""
        bar = make_bar(close=150.0)

        # Patch get_latest_signal to return HOLD
        live_trader.strategy.get_latest_signal = MagicMock(return_value=Signal.HOLD)

        asyncio.get_event_loop().run_until_complete(live_trader.on_bar(bar))

        assert len(live_trader.trade_log) == 0
        assert live_trader._current_position == 0.0

    def test_buy_signal_creates_trade(self, live_trader: LiveTrader) -> None:
        """BUY signal with no position should create a buy trade."""
        bar = make_bar(close=150.0)
        live_trader.strategy.get_latest_signal = MagicMock(return_value=Signal.BUY)

        asyncio.get_event_loop().run_until_complete(live_trader.on_bar(bar))

        assert len(live_trader.trade_log) == 1
        assert live_trader.trade_log[0].side == "buy"
        assert live_trader._current_position > 0.0

    def test_sell_signal_closes_position(self, live_trader: LiveTrader) -> None:
        """SELL signal with open position should close it."""
        bar = make_bar(close=150.0)

        # First open a position
        live_trader.strategy.get_latest_signal = MagicMock(return_value=Signal.BUY)
        asyncio.get_event_loop().run_until_complete(live_trader.on_bar(bar))
        assert live_trader._current_position > 0.0

        # Now close it
        live_trader.strategy.get_latest_signal = MagicMock(return_value=Signal.SELL)
        asyncio.get_event_loop().run_until_complete(live_trader.on_bar(bar))

        assert live_trader._current_position == 0.0
        assert len(live_trader.trade_log) == 2
        assert live_trader.trade_log[1].side == "sell"

    def test_buy_signal_with_existing_position_no_double_buy(
        self, live_trader: LiveTrader
    ) -> None:
        """BUY signal when already long should not create another trade."""
        bar = make_bar(close=150.0)

        # Open position
        live_trader.strategy.get_latest_signal = MagicMock(return_value=Signal.BUY)
        asyncio.get_event_loop().run_until_complete(live_trader.on_bar(bar))
        initial_position = live_trader._current_position

        # Another BUY — should be ignored
        asyncio.get_event_loop().run_until_complete(live_trader.on_bar(bar))

        assert live_trader._current_position == initial_position
        assert len(live_trader.trade_log) == 1  # Still only 1 trade

    def test_bar_count_increments(self, live_trader: LiveTrader) -> None:
        live_trader.strategy.get_latest_signal = MagicMock(return_value=Signal.HOLD)
        for _ in range(5):
            asyncio.get_event_loop().run_until_complete(
                live_trader.on_bar(make_bar())
            )
        assert live_trader._bar_count == 5

    def test_signal_latency_tracked(self, live_trader: LiveTrader) -> None:
        live_trader.strategy.get_latest_signal = MagicMock(return_value=Signal.HOLD)
        asyncio.get_event_loop().run_until_complete(live_trader.on_bar(make_bar()))
        assert len(live_trader._signal_latencies) == 1
        assert live_trader._signal_latencies[0] >= 0.0


# ---------------------------------------------------------------------------
# Risk halt tests
# ---------------------------------------------------------------------------

class TestRiskHalt:
    """Tests for automatic trading halt on risk limit breach."""

    def test_halts_on_drawdown_breach(self, live_trader: LiveTrader) -> None:
        """Trader should stop when drawdown exceeds limit."""
        live_trader.strategy.get_latest_signal = MagicMock(return_value=Signal.HOLD)

        # Manually set metrics to breach drawdown
        live_trader.metrics._peak_equity = 100_000.0
        live_trader.metrics._initialized = True
        live_trader.metrics.current_drawdown = 0.20  # 20% > 15% limit

        asyncio.get_event_loop().run_until_complete(live_trader.on_bar(make_bar()))

        assert live_trader.is_running is False

    def test_does_not_halt_within_limits(self, live_trader: LiveTrader) -> None:
        """Trader should continue when within risk limits."""
        live_trader.is_running = True
        live_trader.strategy.get_latest_signal = MagicMock(return_value=Signal.HOLD)

        # Small drawdown — within limits
        live_trader.metrics._peak_equity = 100_000.0
        live_trader.metrics._initialized = True
        live_trader.metrics.current_drawdown = 0.05  # 5% < 15% limit

        asyncio.get_event_loop().run_until_complete(live_trader.on_bar(make_bar()))

        assert live_trader.is_running is True


# ---------------------------------------------------------------------------
# Trade record tests
# ---------------------------------------------------------------------------

class TestTradeRecord:
    """Tests for TradeRecord dataclass."""

    def test_trade_record_fields(self, live_trader: LiveTrader) -> None:
        bar = make_bar(close=155.0)
        live_trader.strategy.get_latest_signal = MagicMock(return_value=Signal.BUY)

        asyncio.get_event_loop().run_until_complete(live_trader.on_bar(bar))

        assert len(live_trader.trade_log) == 1
        record = live_trader.trade_log[0]
        assert record.symbol == "AAPL"
        assert record.side == "buy"
        assert record.price == 155.0
        assert record.signal == Signal.BUY
        assert record.qty > 0

    def test_sell_record_created_on_close(self, live_trader: LiveTrader) -> None:
        bar = make_bar(close=150.0)

        live_trader.strategy.get_latest_signal = MagicMock(return_value=Signal.BUY)
        asyncio.get_event_loop().run_until_complete(live_trader.on_bar(bar))

        bar2 = make_bar(close=160.0)
        live_trader.strategy.get_latest_signal = MagicMock(return_value=Signal.SELL)
        asyncio.get_event_loop().run_until_complete(live_trader.on_bar(bar2))

        assert live_trader.trade_log[1].side == "sell"
        assert live_trader.trade_log[1].price == 160.0


# ---------------------------------------------------------------------------
# Position sizing tests
# ---------------------------------------------------------------------------

class TestPositionSizing:
    """Tests for _compute_qty() position sizing."""

    def test_qty_based_on_position_size_usd(self, live_trader: LiveTrader) -> None:
        # position_size_usd=10_000, price=100
        # min(10_000, 50_000, 100_000 * 0.5) = min(10_000, 50_000, 50_000) = 10_000
        # qty = int(10_000 / 100) = 100 shares
        qty = live_trader._compute_qty(100.0)
        assert qty == pytest.approx(100.0, abs=1.0)

    def test_zero_price_returns_zero(self, live_trader: LiveTrader) -> None:
        qty = live_trader._compute_qty(0.0)
        assert qty == 0.0

    def test_whole_shares_only(self, live_trader: LiveTrader) -> None:
        qty = live_trader._compute_qty(150.0)
        assert qty == float(int(qty))  # Must be whole number


# ---------------------------------------------------------------------------
# Performance summary tests
# ---------------------------------------------------------------------------

class TestPerformanceSummary:
    """Tests for get_performance_summary()."""

    def test_summary_has_required_keys(self, live_trader: LiveTrader) -> None:
        live_trader.strategy.get_latest_signal = MagicMock(return_value=Signal.HOLD)
        asyncio.get_event_loop().run_until_complete(live_trader.on_bar(make_bar()))

        summary = live_trader.get_performance_summary()
        required_keys = {
            "strategy", "symbol", "n_trades", "bars_processed",
            "current_position", "total_return", "current_drawdown",
        }
        assert required_keys.issubset(set(summary.keys()))

    def test_summary_strategy_name(self, live_trader: LiveTrader) -> None:
        summary = live_trader.get_performance_summary()
        assert "Momentum" in str(summary["strategy"])

    def test_summary_symbol(self, live_trader: LiveTrader) -> None:
        summary = live_trader.get_performance_summary()
        assert summary["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# Simulation loop integration test
# ---------------------------------------------------------------------------

class TestSimulationLoop:
    """Tests for the simulation loop (no API keys needed)."""

    def test_simulation_generates_bars_and_signals(self) -> None:
        """Run simulation for a few bars and verify it processes them."""
        strategy = MomentumStrategy(
            StrategyConfig(params={"short_window": 3, "long_window": 7})
        )
        config = LiveTraderConfig(symbol="TEST", max_drawdown_limit=0.50)
        trader = LiveTrader(strategy, config)

        bars_processed = []

        async def run_limited_simulation() -> None:
            """Run simulation for exactly 20 bars then stop."""
            trader.is_running = True
            strategy.reset_online_state()
            price = 100.0
            np.random.seed(42)

            for _ in range(20):
                ret = np.random.normal(0.0005, 0.015)
                price *= (1 + ret)
                bar = LiveBar(
                    symbol=config.symbol,
                    timestamp=__import__("datetime").datetime.utcnow(),
                    open=price * 0.999,
                    high=price * 1.005,
                    low=price * 0.995,
                    close=price,
                    volume=1_000_000.0,
                )
                await trader.on_bar(bar)
                bars_processed.append(price)

        asyncio.get_event_loop().run_until_complete(run_limited_simulation())

        assert len(bars_processed) == 20
        assert trader._bar_count == 20
        assert len(trader._signal_latencies) == 20
