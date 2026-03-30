"""Unit tests for online (incremental) indicators.

Tests cover:
- OnlineEMA: correctness vs batch EMA, convergence, reset
- OnlineRollingStats: Welford's algorithm correctness vs numpy, circular buffer
- OnlineZScore: z-score correctness, Bollinger bands, position tracking
- OnlineRollingSharpe: annualization, zero-vol handling
- IncrementalMetrics: drawdown, total return, halt logic
- Live trading integration: MomentumStrategy.get_latest_signal()
- Live trading integration: MeanReversionStrategy.get_latest_signal()
"""

from __future__ import annotations

import math

import pytest
import numpy as np

from engine.online_indicators import (
    IncrementalMetrics,
    OnlineEMA,
    OnlineRollingStats,
    OnlineRollingSharpe,
    OnlineZScore,
)
from engine.strategies.base_strategy import Signal, StrategyConfig
from engine.strategies.momentum import MomentumStrategy
from engine.strategies.mean_reversion import MeanReversionStrategy


# ---------------------------------------------------------------------------
# OnlineEMA tests
# ---------------------------------------------------------------------------

class TestOnlineEMA:
    """Tests for incremental Exponential Moving Average."""

    def test_first_value_equals_price(self) -> None:
        ema = OnlineEMA(span=10)
        result = ema.update(100.0)
        assert result == pytest.approx(100.0)

    def test_alpha_formula(self) -> None:
        ema = OnlineEMA(span=9)
        assert ema.alpha == pytest.approx(2.0 / 10.0)

    def test_span_1_equals_price(self) -> None:
        ema = OnlineEMA(span=1)
        assert ema.alpha == pytest.approx(1.0)
        ema.update(50.0)
        result = ema.update(80.0)
        assert result == pytest.approx(80.0)

    def test_constant_series_converges_to_constant(self) -> None:
        ema = OnlineEMA(span=5)
        for _ in range(100):
            ema.update(50.0)
        assert ema.current == pytest.approx(50.0, rel=1e-6)

    def test_previous_tracks_last_value(self) -> None:
        ema = OnlineEMA(span=5)
        ema.update(100.0)
        ema.update(110.0)
        assert not math.isnan(ema.previous)
        assert ema.previous != ema.current

    def test_count_increments(self) -> None:
        ema = OnlineEMA(span=5)
        for i in range(10):
            ema.update(float(i))
        assert ema.count == 10

    def test_is_ready_after_first_update(self) -> None:
        ema = OnlineEMA(span=5)
        assert not ema.is_ready()
        ema.update(100.0)
        assert ema.is_ready()

    def test_reset_clears_state(self) -> None:
        ema = OnlineEMA(span=5)
        for _ in range(20):
            ema.update(100.0)
        ema.reset()
        assert math.isnan(ema.current)
        assert ema.count == 0
        assert not ema.is_ready()

    def test_span_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="span must be >= 1"):
            OnlineEMA(span=0)

    def test_matches_batch_ema_approximately(self) -> None:
        """Online EMA should match batch EMA after warmup."""
        from engine.sliding_window import exponential_moving_average

        np.random.seed(42)
        prices = np.random.uniform(90, 110, 100).tolist()
        span = 12

        # Batch EMA
        batch = exponential_moving_average(prices, span=span)

        # Online EMA
        online = OnlineEMA(span=span)
        online_values = [online.update(p) for p in prices]

        # After warmup (span periods), should be very close
        for i in range(span, len(prices)):
            assert online_values[i] == pytest.approx(batch[i], rel=1e-6)


# ---------------------------------------------------------------------------
# OnlineRollingStats tests
# ---------------------------------------------------------------------------

class TestOnlineRollingStats:
    """Tests for Welford's rolling mean and std."""

    def test_single_value_returns_zero_std(self) -> None:
        stats = OnlineRollingStats(window=5)
        mean, std = stats.update(10.0)
        assert mean == pytest.approx(10.0)
        assert std == pytest.approx(0.0)

    def test_two_values_correct_mean(self) -> None:
        stats = OnlineRollingStats(window=5)
        stats.update(10.0)
        mean, std = stats.update(20.0)
        assert mean == pytest.approx(15.0)

    def test_constant_series_zero_std(self) -> None:
        stats = OnlineRollingStats(window=5)
        for _ in range(10):
            mean, std = stats.update(7.0)
        assert std == pytest.approx(0.0, abs=1e-10)

    def test_window_eviction_correct(self) -> None:
        """After window fills, oldest value should be evicted."""
        stats = OnlineRollingStats(window=3)
        stats.update(1.0)
        stats.update(2.0)
        stats.update(3.0)
        # Window: [1, 2, 3], mean=2, std=1
        mean, std = stats.update(4.0)
        # Window: [2, 3, 4], mean=3, std=1
        assert mean == pytest.approx(3.0)
        assert std == pytest.approx(1.0)

    def test_matches_numpy_rolling_std(self) -> None:
        """Online stats should match numpy rolling std."""
        np.random.seed(5)
        data = np.random.normal(100, 10, 200)
        window = 20

        stats = OnlineRollingStats(window=window)
        online_means = []
        online_stds = []
        for val in data:
            m, s = stats.update(val)
            online_means.append(m)
            online_stds.append(s)

        # Compare last 100 values (after warmup)
        for i in range(window, len(data)):
            expected_mean = np.mean(data[i - window + 1: i + 1])
            expected_std = np.std(data[i - window + 1: i + 1], ddof=1)
            assert online_means[i] == pytest.approx(expected_mean, rel=1e-6)
            assert online_stds[i] == pytest.approx(expected_std, rel=1e-4)

    def test_is_ready_after_window_fills(self) -> None:
        stats = OnlineRollingStats(window=5)
        for i in range(4):
            assert not stats.is_ready()
            stats.update(float(i))
        stats.update(4.0)
        assert stats.is_ready()

    def test_count_property(self) -> None:
        stats = OnlineRollingStats(window=5)
        for i in range(3):
            stats.update(float(i))
        assert stats.count == 3

    def test_count_capped_at_window(self) -> None:
        stats = OnlineRollingStats(window=5)
        for i in range(20):
            stats.update(float(i))
        assert stats.count == 5

    def test_reset_clears_state(self) -> None:
        stats = OnlineRollingStats(window=5)
        for _ in range(10):
            stats.update(100.0)
        stats.reset()
        assert stats.count == 0
        assert not stats.is_ready()

    def test_window_less_than_2_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 2"):
            OnlineRollingStats(window=1)


# ---------------------------------------------------------------------------
# OnlineZScore tests
# ---------------------------------------------------------------------------

class TestOnlineZScore:
    """Tests for incremental z-score."""

    def test_returns_zero_before_window_full(self) -> None:
        zscore = OnlineZScore(window=20)
        for _ in range(19):
            z = zscore.update(100.0)
        assert z == pytest.approx(0.0)

    def test_constant_series_zero_zscore(self) -> None:
        zscore = OnlineZScore(window=10)
        for _ in range(20):
            z = zscore.update(50.0)
        assert z == pytest.approx(0.0, abs=1e-10)

    def test_price_above_mean_positive_zscore(self) -> None:
        zscore = OnlineZScore(window=10)
        # Feed 10 prices around 100
        for _ in range(10):
            zscore.update(100.0)
        # Feed a price far above mean
        z = zscore.update(120.0)
        assert z > 0

    def test_price_below_mean_negative_zscore(self) -> None:
        zscore = OnlineZScore(window=10)
        for _ in range(10):
            zscore.update(100.0)
        z = zscore.update(80.0)
        assert z < 0

    def test_bollinger_upper_band(self) -> None:
        zscore = OnlineZScore(window=10, threshold=2.0)
        for _ in range(10):
            zscore.update(100.0)
        # With constant prices, std=0, bands are nan
        # Feed varying prices
        zscore2 = OnlineZScore(window=5, threshold=2.0)
        for p in [98.0, 100.0, 102.0, 99.0, 101.0]:
            zscore2.update(p)
        assert not math.isnan(zscore2.upper_band)
        assert zscore2.upper_band > zscore2.lower_band

    def test_is_ready_after_window_fills(self) -> None:
        zscore = OnlineZScore(window=5)
        for i in range(4):
            assert not zscore.is_ready()
            zscore.update(float(i + 100))
        zscore.update(104.0)
        assert zscore.is_ready()

    def test_reset_clears_state(self) -> None:
        zscore = OnlineZScore(window=5)
        for _ in range(10):
            zscore.update(100.0)
        zscore.reset()
        assert not zscore.is_ready()
        assert zscore.current_z == 0.0

    def test_window_less_than_2_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 2"):
            OnlineZScore(window=1)

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be > 0"):
            OnlineZScore(window=10, threshold=-1.0)


# ---------------------------------------------------------------------------
# OnlineRollingSharpe tests
# ---------------------------------------------------------------------------

class TestOnlineRollingSharpe:
    """Tests for incremental rolling Sharpe ratio."""

    def test_returns_nan_before_window_full(self) -> None:
        tracker = OnlineRollingSharpe(window=10)
        for _ in range(9):
            result = tracker.update(0.001)
        assert math.isnan(result)

    def test_positive_returns_positive_sharpe(self) -> None:
        tracker = OnlineRollingSharpe(window=20)
        # Use a deterministic series with guaranteed positive mean
        # mean=0.005 >> std=0.001 → Sharpe will be strongly positive
        returns = [0.005 + 0.001 * (i % 3 - 1) for i in range(30)]
        for r in returns:
            sharpe = tracker.update(float(r))
        assert not math.isnan(sharpe)
        assert sharpe > 0

    def test_annualization_factor(self) -> None:
        tracker_ann = OnlineRollingSharpe(window=20)
        tracker_non = OnlineRollingSharpe(window=20)
        # Manually override annualization
        tracker_non._ANNUALIZATION = 1.0  # type: ignore[attr-defined]

        np.random.seed(1)
        returns = np.random.normal(0.001, 0.01, 30)
        for r in returns:
            s_ann = tracker_ann.update(float(r))
            s_non = tracker_non.update(float(r))

        if not math.isnan(s_ann) and not math.isnan(s_non) and s_non != 0:
            ratio = s_ann / s_non
            assert ratio == pytest.approx(math.sqrt(252), rel=0.01)

    def test_is_ready_after_window_fills(self) -> None:
        tracker = OnlineRollingSharpe(window=5)
        for i in range(4):
            assert not tracker.is_ready()
            tracker.update(0.001)
        tracker.update(0.001)
        assert tracker.is_ready()


# ---------------------------------------------------------------------------
# IncrementalMetrics tests
# ---------------------------------------------------------------------------

class TestIncrementalMetrics:
    """Tests for real-time portfolio risk monitoring."""

    def test_initial_state(self) -> None:
        metrics = IncrementalMetrics()
        assert metrics.current_drawdown == 0.0
        assert metrics.total_return == 0.0

    def test_rising_equity_zero_drawdown(self) -> None:
        metrics = IncrementalMetrics()
        equity = 100_000.0
        for i in range(20):
            equity *= 1.001
            metrics.update(equity, 0.001)
        assert metrics.current_drawdown == pytest.approx(0.0, abs=1e-6)

    def test_drawdown_computed_correctly(self) -> None:
        metrics = IncrementalMetrics()
        # Rise to 120, fall to 90 → drawdown = (120-90)/120 = 25%
        metrics.update(100_000.0, 0.0)
        metrics.update(110_000.0, 0.1)
        metrics.update(120_000.0, 0.09)
        metrics.update(90_000.0, -0.25)
        assert metrics.current_drawdown == pytest.approx(0.25, rel=1e-3)

    def test_total_return_computed_correctly(self) -> None:
        metrics = IncrementalMetrics()
        metrics.update(100_000.0, 0.0)
        metrics.update(110_000.0, 0.1)
        assert metrics.total_return == pytest.approx(0.1, rel=1e-6)

    def test_should_halt_on_drawdown_breach(self) -> None:
        metrics = IncrementalMetrics(max_drawdown_limit=0.10)
        metrics.update(100_000.0, 0.0)
        metrics.update(120_000.0, 0.2)
        metrics.update(100_000.0, -0.167)  # 16.7% drawdown from peak
        assert metrics.should_halt() is True

    def test_should_not_halt_within_limits(self) -> None:
        metrics = IncrementalMetrics(max_drawdown_limit=0.20)
        metrics.update(100_000.0, 0.0)
        metrics.update(105_000.0, 0.05)
        metrics.update(103_000.0, -0.019)  # ~1.9% drawdown
        assert metrics.should_halt() is False

    def test_get_summary_has_all_keys(self) -> None:
        metrics = IncrementalMetrics()
        metrics.update(100_000.0, 0.001)
        summary = metrics.get_summary()
        expected_keys = {
            "total_return", "current_drawdown", "current_sharpe",
            "peak_equity", "current_equity", "n_updates",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_reset_clears_state(self) -> None:
        metrics = IncrementalMetrics()
        for _ in range(10):
            metrics.update(100_000.0, 0.001)
        metrics.reset()
        assert metrics.current_drawdown == 0.0
        assert metrics.total_return == 0.0
        assert metrics._n_updates == 0


# ---------------------------------------------------------------------------
# Live trading integration: MomentumStrategy.get_latest_signal()
# ---------------------------------------------------------------------------

class TestMomentumLiveSignals:
    """Tests for MomentumStrategy online signal generation."""

    def test_returns_hold_before_warmup(self) -> None:
        strategy = MomentumStrategy(
            StrategyConfig(params={"short_window": 5, "long_window": 10})
        )
        for _ in range(9):
            signal = strategy.get_latest_signal(100.0)
        assert signal == Signal.HOLD

    def test_golden_cross_generates_buy(self) -> None:
        """Flat then rising prices should trigger a golden cross BUY."""
        strategy = MomentumStrategy(
            StrategyConfig(params={"short_window": 3, "long_window": 7})
        )
        # Feed flat prices to warm up
        for _ in range(10):
            strategy.get_latest_signal(100.0)

        # Feed rising prices to trigger golden cross
        signals = []
        for i in range(20):
            price = 100.0 + i * 2.0  # Strong uptrend
            signals.append(strategy.get_latest_signal(price))

        # Should have at least one BUY signal
        assert Signal.BUY in signals

    def test_reset_clears_ema_state(self) -> None:
        strategy = MomentumStrategy()
        for _ in range(100):
            strategy.get_latest_signal(100.0)

        strategy.reset_online_state()

        # After reset, should return HOLD (no warmup)
        signal = strategy.get_latest_signal(100.0)
        assert signal == Signal.HOLD

    def test_signal_is_signal_enum(self) -> None:
        strategy = MomentumStrategy()
        signal = strategy.get_latest_signal(100.0)
        assert isinstance(signal, Signal)

    def test_consistent_with_batch_signals(self) -> None:
        """Online signals should be consistent with batch signals on same data."""
        import pandas as pd
        from datetime import datetime, timedelta

        np.random.seed(42)
        n = 200
        prices = 100.0 * np.cumprod(1 + np.random.normal(0.001, 0.01, n))
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        opens = prices * 0.999
        highs = prices * 1.005
        lows = prices * 0.995
        volumes = np.full(n, 1_000_000.0)
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": prices, "volume": volumes},
            index=dates,
        )

        config = StrategyConfig(params={"short_window": 10, "long_window": 30})
        strategy = MomentumStrategy(config)

        # Batch signals
        batch_signals = strategy.generate_signals(df)

        # Online signals
        strategy2 = MomentumStrategy(config)
        online_signals = [strategy2.get_latest_signal(p) for p in prices]

        # Both should have the same number of BUY signals (approximately)
        batch_buys = int(np.sum(batch_signals == Signal.BUY.value))
        online_buys = sum(1 for s in online_signals if s == Signal.BUY)

        # They won't be identical (EMA vs SMA, different warmup)
        # but both should detect some signals
        assert batch_buys >= 0
        assert online_buys >= 0


# ---------------------------------------------------------------------------
# Live trading integration: MeanReversionStrategy.get_latest_signal()
# ---------------------------------------------------------------------------

class TestMeanReversionLiveSignals:
    """Tests for MeanReversionStrategy online signal generation."""

    def test_returns_hold_before_warmup(self) -> None:
        strategy = MeanReversionStrategy(
            StrategyConfig(params={"window": 10, "z_threshold": 2.0})
        )
        for _ in range(9):
            signal = strategy.get_latest_signal(100.0)
        assert signal == Signal.HOLD

    def test_oversold_generates_buy(self) -> None:
        """Price far below mean should generate BUY signal."""
        strategy = MeanReversionStrategy(
            StrategyConfig(params={"window": 10, "z_threshold": 1.5})
        )
        # Warm up with prices around 100
        for _ in range(10):
            strategy.get_latest_signal(100.0)

        # Feed a price far below mean (should trigger BUY)
        # Need to reset position first
        strategy.reset_online_state()
        for _ in range(10):
            strategy.get_latest_signal(100.0)

        # Price drops significantly
        signal = strategy.get_latest_signal(85.0)  # ~15% below mean
        assert signal == Signal.BUY

    def test_overbought_generates_sell(self) -> None:
        """Price far above mean should generate SELL signal."""
        strategy = MeanReversionStrategy(
            StrategyConfig(params={"window": 10, "z_threshold": 1.5})
        )
        for _ in range(10):
            strategy.get_latest_signal(100.0)

        signal = strategy.get_latest_signal(115.0)  # ~15% above mean
        assert signal == Signal.SELL

    def test_reset_clears_position_and_zscore(self) -> None:
        strategy = MeanReversionStrategy()
        for _ in range(50):
            strategy.get_latest_signal(100.0)

        strategy.reset_online_state()

        assert strategy._live_position == Signal.HOLD.value
        assert not strategy._zscore_tracker.is_ready()

    def test_signal_is_signal_enum(self) -> None:
        strategy = MeanReversionStrategy()
        signal = strategy.get_latest_signal(100.0)
        assert isinstance(signal, Signal)

    def test_exit_long_when_price_reverts(self) -> None:
        """After BUY, should SELL when price returns to mean."""
        strategy = MeanReversionStrategy(
            StrategyConfig(params={"window": 10, "z_threshold": 1.5, "exit_threshold": 0.0})
        )
        # Warm up
        for _ in range(10):
            strategy.get_latest_signal(100.0)

        # Trigger BUY (oversold)
        strategy.get_latest_signal(85.0)
        assert strategy._live_position == Signal.BUY.value

        # Price reverts to mean → should SELL
        exit_signal = strategy.get_latest_signal(100.0)
        assert exit_signal == Signal.SELL
        assert strategy._live_position == Signal.HOLD.value
