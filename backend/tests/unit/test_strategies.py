"""Unit tests for trading strategy implementations.

Tests cover:
- BaseStrategy: abstract enforcement, config validation, DataFrame validation
- StrategyConfig: validation of all fields
- MomentumStrategy: signal generation, crossover detection, param validation
- MeanReversionStrategy: z-score signals, Bollinger bands, position tracking
- Signal enum values
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from engine.strategies.base_strategy import (
    BaseStrategy,
    BacktestResult,
    Signal,
    StrategyConfig,
)
from engine.strategies.momentum import MomentumStrategy
from engine.strategies.mean_reversion import MeanReversionStrategy


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def make_ohlcv_df(
    n: int = 200,
    start_price: float = 100.0,
    trend: float = 0.001,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for testing.

    Args:
        n: Number of rows (trading days).
        start_price: Starting close price.
        trend: Daily drift (positive = uptrend).
        volatility: Daily return std.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: open, high, low, close, volume.
    """
    np.random.seed(seed)
    returns = np.random.normal(trend, volatility, n)
    closes = start_price * np.cumprod(1 + returns)
    opens = closes * np.random.uniform(0.995, 1.005, n)
    highs = np.maximum(opens, closes) * np.random.uniform(1.001, 1.015, n)
    lows = np.minimum(opens, closes) * np.random.uniform(0.985, 0.999, n)
    volumes = np.random.randint(1_000_000, 10_000_000, n).astype(float)

    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )


def make_trending_df(n: int = 200) -> pd.DataFrame:
    """Generate a strongly uptrending OHLCV DataFrame."""
    return make_ohlcv_df(n=n, trend=0.005, volatility=0.005, seed=1)


def make_mean_reverting_df(n: int = 200) -> pd.DataFrame:
    """Generate a mean-reverting (oscillating) OHLCV DataFrame."""
    np.random.seed(99)
    # Oscillate around 100
    closes = 100.0 + 10.0 * np.sin(np.linspace(0, 8 * np.pi, n))
    closes += np.random.normal(0, 0.5, n)
    opens = closes * np.random.uniform(0.998, 1.002, n)
    highs = np.maximum(opens, closes) * 1.005
    lows = np.minimum(opens, closes) * 0.995
    volumes = np.full(n, 1_000_000.0)
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )


@pytest.fixture
def default_df() -> pd.DataFrame:
    return make_ohlcv_df()


@pytest.fixture
def trending_df() -> pd.DataFrame:
    return make_trending_df()


@pytest.fixture
def mean_reverting_df() -> pd.DataFrame:
    return make_mean_reverting_df()


# ---------------------------------------------------------------------------
# Signal enum tests
# ---------------------------------------------------------------------------

class TestSignalEnum:
    """Tests for Signal enum values."""

    def test_buy_value(self) -> None:
        assert Signal.BUY.value == 1

    def test_hold_value(self) -> None:
        assert Signal.HOLD.value == 0

    def test_sell_value(self) -> None:
        assert Signal.SELL.value == -1

    def test_signal_comparison(self) -> None:
        assert Signal.BUY > Signal.HOLD
        assert Signal.HOLD > Signal.SELL


# ---------------------------------------------------------------------------
# StrategyConfig tests
# ---------------------------------------------------------------------------

class TestStrategyConfig:
    """Tests for StrategyConfig validation."""

    def test_default_values(self) -> None:
        config = StrategyConfig()
        assert config.initial_capital == 100_000.0
        assert config.position_size == 1.0
        assert config.stop_loss == 0.02
        assert config.commission == 0.001

    def test_custom_values(self) -> None:
        config = StrategyConfig(
            initial_capital=50_000.0,
            position_size=0.5,
            commission=0.002,
        )
        assert config.initial_capital == 50_000.0
        assert config.position_size == 0.5

    def test_zero_capital_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_capital"):
            StrategyConfig(initial_capital=0.0)

    def test_negative_capital_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_capital"):
            StrategyConfig(initial_capital=-1000.0)

    def test_position_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="position_size"):
            StrategyConfig(position_size=0.0)

    def test_position_size_over_1_raises(self) -> None:
        with pytest.raises(ValueError, match="position_size"):
            StrategyConfig(position_size=1.5)

    def test_negative_stop_loss_raises(self) -> None:
        with pytest.raises(ValueError, match="stop_loss"):
            StrategyConfig(stop_loss=-0.01)

    def test_high_commission_raises(self) -> None:
        with pytest.raises(ValueError, match="commission"):
            StrategyConfig(commission=0.5)

    def test_params_default_empty_dict(self) -> None:
        config = StrategyConfig()
        assert config.params == {}


# ---------------------------------------------------------------------------
# BaseStrategy abstract enforcement tests
# ---------------------------------------------------------------------------

class TestBaseStrategyAbstract:
    """Tests that BaseStrategy cannot be instantiated directly."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            BaseStrategy()  # type: ignore[abstract]

    def test_subclass_without_generate_signals_raises(self) -> None:
        class IncompleteStrategy(BaseStrategy):
            def get_name(self) -> str:
                return "Incomplete"
            # Missing generate_signals

        with pytest.raises(TypeError):
            IncompleteStrategy()  # type: ignore[abstract]

    def test_subclass_without_get_name_raises(self) -> None:
        class IncompleteStrategy(BaseStrategy):
            def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
                return np.zeros(len(df), dtype=int)
            # Missing get_name

        with pytest.raises(TypeError):
            IncompleteStrategy()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# DataFrame validation tests (via MomentumStrategy)
# ---------------------------------------------------------------------------

class TestDataFrameValidation:
    """Tests for _validate_dataframe() in BaseStrategy."""

    def test_missing_close_column_raises(self) -> None:
        strategy = MomentumStrategy()
        df = make_ohlcv_df()
        df = df.drop(columns=["close"])
        with pytest.raises(ValueError, match="missing required columns"):
            strategy.generate_signals(df)

    def test_empty_dataframe_raises(self) -> None:
        strategy = MomentumStrategy()
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with pytest.raises(ValueError, match="empty"):
            strategy.generate_signals(df)

    def test_single_row_raises(self) -> None:
        strategy = MomentumStrategy()
        df = make_ohlcv_df(n=1)
        with pytest.raises(ValueError):
            strategy.generate_signals(df)


# ---------------------------------------------------------------------------
# MomentumStrategy tests
# ---------------------------------------------------------------------------

class TestMomentumStrategy:
    """Tests for MomentumStrategy signal generation."""

    def test_default_name(self) -> None:
        strategy = MomentumStrategy()
        assert "Momentum" in strategy.get_name()
        assert "SMA" in strategy.get_name()
        assert "20" in strategy.get_name()
        assert "50" in strategy.get_name()

    def test_ema_name(self) -> None:
        config = StrategyConfig(params={"use_ema": 1})
        strategy = MomentumStrategy(config)
        assert "EMA" in strategy.get_name()

    def test_custom_windows_in_name(self) -> None:
        config = StrategyConfig(params={"short_window": 10, "long_window": 30})
        strategy = MomentumStrategy(config)
        assert "10" in strategy.get_name()
        assert "30" in strategy.get_name()

    def test_short_window_gte_long_raises(self) -> None:
        with pytest.raises(ValueError, match="short_window.*<.*long_window"):
            MomentumStrategy(StrategyConfig(params={"short_window": 50, "long_window": 20}))

    def test_short_window_equals_long_raises(self) -> None:
        with pytest.raises(ValueError, match="short_window.*<.*long_window"):
            MomentumStrategy(StrategyConfig(params={"short_window": 20, "long_window": 20}))

    def test_signals_correct_length(self, default_df: pd.DataFrame) -> None:
        strategy = MomentumStrategy()
        signals = strategy.generate_signals(default_df)
        assert len(signals) == len(default_df)

    def test_signals_only_valid_values(self, default_df: pd.DataFrame) -> None:
        strategy = MomentumStrategy()
        signals = strategy.generate_signals(default_df)
        valid_values = {Signal.BUY.value, Signal.HOLD.value, Signal.SELL.value}
        assert set(np.unique(signals)).issubset(valid_values)

    def test_signals_are_integers(self, default_df: pd.DataFrame) -> None:
        strategy = MomentumStrategy()
        signals = strategy.generate_signals(default_df)
        assert signals.dtype in [np.int32, np.int64, int]

    def test_trending_market_generates_buy_signals(self) -> None:
        # Build a price series that starts flat then trends up sharply,
        # guaranteeing a golden cross (short MA crosses above long MA).
        np.random.seed(10)
        # First 50 bars: flat around 100
        flat = np.full(50, 100.0) + np.random.normal(0, 0.1, 50)
        # Next 150 bars: strong uptrend
        trend = 100.0 + np.arange(1, 151) * 0.5 + np.random.normal(0, 0.2, 150)
        closes = np.concatenate([flat, trend])
        opens = closes * np.random.uniform(0.999, 1.001, len(closes))
        highs = closes * 1.005
        lows = closes * 0.995
        volumes = np.full(len(closes), 1_000_000.0)
        dates = pd.date_range("2020-01-01", periods=len(closes), freq="B")
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
            index=dates,
        )
        strategy = MomentumStrategy(
            StrategyConfig(params={"short_window": 10, "long_window": 30})
        )
        signals = strategy.generate_signals(df)
        n_buys = int(np.sum(signals == Signal.BUY.value))
        assert n_buys >= 1, "Flat-then-trending market must generate at least one BUY signal"

    def test_too_short_dataframe_raises(self) -> None:
        strategy = MomentumStrategy()  # long_window=50
        df = make_ohlcv_df(n=40)  # Only 40 rows, need 51
        with pytest.raises(ValueError, match="requires at least"):
            strategy.generate_signals(df)

    def test_ema_mode_generates_signals(self, default_df: pd.DataFrame) -> None:
        config = StrategyConfig(params={"short_window": 10, "long_window": 30, "use_ema": 1})
        strategy = MomentumStrategy(config)
        signals = strategy.generate_signals(default_df)
        assert len(signals) == len(default_df)

    def test_repr_contains_name(self) -> None:
        strategy = MomentumStrategy()
        assert "MomentumStrategy" in repr(strategy)

    def test_first_long_window_signals_are_hold(self, default_df: pd.DataFrame) -> None:
        strategy = MomentumStrategy()  # long_window=50
        signals = strategy.generate_signals(default_df)
        # First 50 signals must be HOLD (no MA available yet)
        assert all(s == Signal.HOLD.value for s in signals[:50])


# ---------------------------------------------------------------------------
# MeanReversionStrategy tests
# ---------------------------------------------------------------------------

class TestMeanReversionStrategy:
    """Tests for MeanReversionStrategy signal generation."""

    def test_default_name(self) -> None:
        strategy = MeanReversionStrategy()
        assert "MeanReversion" in strategy.get_name()
        assert "W20" in strategy.get_name()
        assert "Z2.0" in strategy.get_name()

    def test_custom_params_in_name(self) -> None:
        config = StrategyConfig(params={"window": 30, "z_threshold": 1.5})
        strategy = MeanReversionStrategy(config)
        assert "W30" in strategy.get_name()
        assert "Z1.5" in strategy.get_name()

    def test_window_less_than_2_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 2"):
            MeanReversionStrategy(StrategyConfig(params={"window": 1}))

    def test_negative_z_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be > 0"):
            MeanReversionStrategy(StrategyConfig(params={"z_threshold": -1.0}))

    def test_zero_z_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be > 0"):
            MeanReversionStrategy(StrategyConfig(params={"z_threshold": 0.0}))

    def test_signals_correct_length(self, default_df: pd.DataFrame) -> None:
        strategy = MeanReversionStrategy()
        signals = strategy.generate_signals(default_df)
        assert len(signals) == len(default_df)

    def test_signals_only_valid_values(self, default_df: pd.DataFrame) -> None:
        strategy = MeanReversionStrategy()
        signals = strategy.generate_signals(default_df)
        valid_values = {Signal.BUY.value, Signal.HOLD.value, Signal.SELL.value}
        assert set(np.unique(signals)).issubset(valid_values)

    def test_mean_reverting_market_generates_signals(
        self, mean_reverting_df: pd.DataFrame
    ) -> None:
        config = StrategyConfig(params={"window": 20, "z_threshold": 1.5})
        strategy = MeanReversionStrategy(config)
        signals = strategy.generate_signals(mean_reverting_df)
        n_buys = int(np.sum(signals == Signal.BUY.value))
        n_sells = int(np.sum(signals == Signal.SELL.value))
        assert n_buys >= 1, "Mean-reverting market should generate BUY signals"
        assert n_sells >= 1, "Mean-reverting market should generate SELL signals"

    def test_too_short_dataframe_raises(self) -> None:
        strategy = MeanReversionStrategy()  # window=20
        df = make_ohlcv_df(n=15)
        with pytest.raises(ValueError, match="requires at least"):
            strategy.generate_signals(df)

    def test_bollinger_bands_shape(self, default_df: pd.DataFrame) -> None:
        strategy = MeanReversionStrategy()
        upper, mid, lower = strategy.compute_bollinger_bands(default_df)
        assert len(upper) == len(default_df)
        assert len(mid) == len(default_df)
        assert len(lower) == len(default_df)

    def test_bollinger_bands_ordering(self, default_df: pd.DataFrame) -> None:
        strategy = MeanReversionStrategy()
        upper, mid, lower = strategy.compute_bollinger_bands(default_df)
        # Where not NaN: upper >= mid >= lower
        valid = ~np.isnan(upper) & ~np.isnan(mid) & ~np.isnan(lower)
        assert np.all(upper[valid] >= mid[valid])
        assert np.all(mid[valid] >= lower[valid])

    def test_lower_z_threshold_generates_more_signals(
        self, default_df: pd.DataFrame
    ) -> None:
        config_tight = StrategyConfig(params={"window": 20, "z_threshold": 1.0})
        config_wide = StrategyConfig(params={"window": 20, "z_threshold": 3.0})
        strategy_tight = MeanReversionStrategy(config_tight)
        strategy_wide = MeanReversionStrategy(config_wide)

        signals_tight = strategy_tight.generate_signals(default_df)
        signals_wide = strategy_wide.generate_signals(default_df)

        n_signals_tight = int(np.sum(signals_tight != Signal.HOLD.value))
        n_signals_wide = int(np.sum(signals_wide != Signal.HOLD.value))

        # Tighter threshold → more signals triggered
        assert n_signals_tight >= n_signals_wide

    def test_first_window_signals_are_hold(self, default_df: pd.DataFrame) -> None:
        strategy = MeanReversionStrategy()  # window=20
        signals = strategy.generate_signals(default_df)
        # First 20 signals must be HOLD (no rolling stats yet)
        assert all(s == Signal.HOLD.value for s in signals[:20])
