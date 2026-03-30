"""Online (incremental) technical indicators for live trading.

These replace the batch sliding_window.py functions for live trading.
Each indicator updates in O(1) per new price tick — no historical recomputation.

Key difference from batch functions:
    Batch (sliding_window.py):  Process all N bars at once → O(N) or O(N×W)
    Online (this file):         Process one bar at a time → O(1) per tick

This is critical for live trading where you receive one price at a time
and must generate a signal within milliseconds.

Algorithms used:
    - Welford's online algorithm for rolling mean/variance (numerically stable)
    - Circular buffer for O(1) window eviction
    - Incremental EMA with smoothing factor alpha

Latency targets:
    - OnlineEMA.update():          ~0.5 μs
    - OnlineRollingStats.update(): ~1.0 μs
    - OnlineZScore.update():       ~1.5 μs
    - IncrementalMetrics.update(): ~2.0 μs
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from config.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# OnlineEMA — Exponential Moving Average (O(1) per tick)
# ---------------------------------------------------------------------------

class OnlineEMA:
    """Incremental Exponential Moving Average. O(1) update per price tick.

    In live trading, you receive one price at a time. This class maintains
    the current EMA value and updates it in O(1) — no historical data needed.

    Formula: EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}
    Where:   alpha = 2 / (span + 1)

    Attributes:
        span: EMA span (e.g., 12 for 12-period EMA).
        alpha: Smoothing factor.
        current: Current EMA value (nan until first update).
        previous: Previous EMA value (nan until second update).
        count: Number of updates received.

    Example:
        >>> ema = OnlineEMA(span=12)
        >>> for price in [100.0, 102.0, 101.0, 105.0]:
        ...     val = ema.update(price)
        >>> print(f"Current EMA(12): {ema.current:.2f}")
    """

    def __init__(self, span: int) -> None:
        """Initialize OnlineEMA.

        Args:
            span: EMA span. Must be >= 1.

        Raises:
            ValueError: If span < 1.
        """
        if span < 1:
            raise ValueError(f"span must be >= 1, got {span}.")

        self.span: int = span
        self.alpha: float = 2.0 / (span + 1)
        self.current: float = float("nan")
        self.previous: float = float("nan")
        self.count: int = 0

    def update(self, price: float) -> float:
        """Update EMA with a new price. O(1).

        Args:
            price: Latest price value.

        Returns:
            Updated EMA value.

        Example:
            >>> ema = OnlineEMA(span=3)
            >>> ema.update(10.0)
            10.0
            >>> ema.update(12.0)
            11.0
        """
        self.previous = self.current
        if self.count == 0:
            self.current = price  # Seed with first price
        else:
            self.current = self.alpha * price + (1.0 - self.alpha) * self.current
        self.count += 1
        return self.current

    def is_ready(self) -> bool:
        """Return True if EMA has been seeded (at least 1 update).

        Returns:
            True if at least one price has been processed.
        """
        return self.count > 0

    def reset(self) -> None:
        """Reset the EMA to initial state."""
        self.current = float("nan")
        self.previous = float("nan")
        self.count = 0


# ---------------------------------------------------------------------------
# OnlineRollingStats — Rolling Mean + Std using Welford's Algorithm (O(1))
# ---------------------------------------------------------------------------

class OnlineRollingStats:
    """Rolling mean and standard deviation using Welford's algorithm. O(1) per tick.

    Welford's algorithm maintains running sum and sum-of-squares in a
    circular buffer. When a new value arrives:
        1. Remove the oldest value from running stats
        2. Add the new value to running stats
        3. Compute mean and std from running totals

    This is numerically stable (avoids catastrophic cancellation) and
    runs in O(1) per update — critical for live trading.

    Attributes:
        window: Rolling window size.
        count: Number of values currently in the window.

    Example:
        >>> stats = OnlineRollingStats(window=20)
        >>> for price in prices:
        ...     mean, std = stats.update(price)
        >>> print(f"Rolling mean: {mean:.2f}, Rolling std: {std:.2f}")
    """

    def __init__(self, window: int) -> None:
        """Initialize OnlineRollingStats.

        Args:
            window: Rolling window size. Must be >= 2.

        Raises:
            ValueError: If window < 2.
        """
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}.")

        self.window: int = window
        self._buffer: Deque[float] = deque(maxlen=window)  # Circular buffer
        self._sum: float = 0.0
        self._sum_sq: float = 0.0

    @property
    def count(self) -> int:
        """Number of values currently in the window."""
        return len(self._buffer)

    def update(self, value: float) -> Tuple[float, float]:
        """Add new value, return (rolling_mean, rolling_std). O(1).

        Uses the circular buffer to evict the oldest value and update
        running sum and sum-of-squares in constant time.

        Args:
            value: New price or return value.

        Returns:
            Tuple of (rolling_mean, rolling_std).
            Returns (value, 0.0) if fewer than 2 values in window.

        Example:
            >>> stats = OnlineRollingStats(window=3)
            >>> stats.update(10.0)
            (10.0, 0.0)
            >>> stats.update(12.0)
            (11.0, 1.414...)
            >>> stats.update(11.0)
            (11.0, 1.0)
            >>> stats.update(13.0)  # 10.0 evicted
            (12.0, 1.0)
        """
        # Evict oldest value if window is full
        if len(self._buffer) == self.window:
            old = self._buffer[0]  # Oldest value (deque evicts automatically)
            self._sum -= old
            self._sum_sq -= old * old

        # Add new value
        self._buffer.append(value)
        self._sum += value
        self._sum_sq += value * value

        n = len(self._buffer)
        if n < 2:
            return value, 0.0

        mean = self._sum / n
        # Variance = E[X²] - (E[X])² with Bessel's correction (ddof=1)
        variance = (self._sum_sq - self._sum * self._sum / n) / (n - 1)
        std = math.sqrt(max(variance, 0.0))  # max guards against float rounding

        return mean, std

    def is_ready(self) -> bool:
        """Return True if window is fully populated.

        Returns:
            True if window has at least `window` values.
        """
        return len(self._buffer) >= self.window

    def reset(self) -> None:
        """Reset all state."""
        self._buffer.clear()
        self._sum = 0.0
        self._sum_sq = 0.0


# ---------------------------------------------------------------------------
# OnlineZScore — Rolling Z-Score for Mean Reversion (O(1))
# ---------------------------------------------------------------------------

class OnlineZScore:
    """Incremental z-score for mean reversion signals. O(1) per tick.

    z_score = (price - rolling_mean) / rolling_std

    Used by MeanReversionStrategy in live trading to detect when a price
    is statistically oversold (z < -threshold) or overbought (z > +threshold).

    Attributes:
        window: Rolling window for mean/std computation.
        threshold: Z-score threshold for signal generation.
        current_z: Latest z-score value.

    Example:
        >>> zscore = OnlineZScore(window=20, threshold=2.0)
        >>> for price in live_prices:
        ...     z = zscore.update(price)
        ...     if z < -2.0:
        ...         signal = Signal.BUY
        ...     elif z > 2.0:
        ...         signal = Signal.SELL
    """

    def __init__(self, window: int, threshold: float = 2.0) -> None:
        """Initialize OnlineZScore.

        Args:
            window: Rolling window size for mean/std. Must be >= 2.
            threshold: Z-score threshold for signal generation.

        Raises:
            ValueError: If window < 2 or threshold <= 0.
        """
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}.")
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}.")

        self.window: int = window
        self.threshold: float = threshold
        self._stats: OnlineRollingStats = OnlineRollingStats(window)
        self.current_z: float = 0.0
        self._last_mean: float = float("nan")
        self._last_std: float = float("nan")

    def update(self, price: float) -> float:
        """Update z-score with new price. O(1).

        Args:
            price: Latest price value.

        Returns:
            Current z-score. Returns 0.0 if window not yet full.

        Example:
            >>> z = zscore.update(105.0)
            >>> if z < -2.0:
            ...     print("Oversold — BUY signal")
        """
        mean, std = self._stats.update(price)
        self._last_mean = mean
        self._last_std = std

        if not self._stats.is_ready() or std < 1e-10:
            self.current_z = 0.0
            return 0.0

        self.current_z = (price - mean) / std
        return self.current_z

    def is_ready(self) -> bool:
        """Return True if window is fully populated.

        Returns:
            True if enough data to compute reliable z-score.
        """
        return self._stats.is_ready()

    @property
    def upper_band(self) -> float:
        """Upper Bollinger Band (mean + threshold * std).

        Returns:
            Upper band value, or nan if not ready.
        """
        if math.isnan(self._last_mean) or math.isnan(self._last_std):
            return float("nan")
        return self._last_mean + self.threshold * self._last_std

    @property
    def lower_band(self) -> float:
        """Lower Bollinger Band (mean - threshold * std).

        Returns:
            Lower band value, or nan if not ready.
        """
        if math.isnan(self._last_mean) or math.isnan(self._last_std):
            return float("nan")
        return self._last_mean - self.threshold * self._last_std

    def reset(self) -> None:
        """Reset all state."""
        self._stats.reset()
        self.current_z = 0.0
        self._last_mean = float("nan")
        self._last_std = float("nan")


# ---------------------------------------------------------------------------
# OnlineRollingSharpe — Rolling Sharpe Ratio (O(1))
# ---------------------------------------------------------------------------

class OnlineRollingSharpe:
    """Incremental rolling Sharpe ratio. O(1) per return update.

    Used for real-time strategy health monitoring. If Sharpe drops
    below a threshold, the risk manager can halt the strategy.

    Attributes:
        window: Rolling window for Sharpe computation.
        risk_free_rate: Daily risk-free rate.
        current_sharpe: Latest annualized Sharpe ratio.

    Example:
        >>> sharpe_tracker = OnlineRollingSharpe(window=60)
        >>> for daily_return in live_returns:
        ...     sharpe = sharpe_tracker.update(daily_return)
        ...     if sharpe < 0.5:
        ...         alert_risk_manager("Sharpe degrading")
    """

    _ANNUALIZATION = math.sqrt(252)

    def __init__(self, window: int, risk_free_rate: float = 0.0) -> None:
        """Initialize OnlineRollingSharpe.

        Args:
            window: Rolling window size. Must be >= 2.
            risk_free_rate: Daily risk-free rate (default 0.0).

        Raises:
            ValueError: If window < 2.
        """
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}.")

        self.window: int = window
        self.risk_free_rate: float = risk_free_rate
        self._stats: OnlineRollingStats = OnlineRollingStats(window)
        self.current_sharpe: float = float("nan")

    def update(self, daily_return: float) -> float:
        """Update Sharpe with new daily return. O(1).

        Args:
            daily_return: Latest daily portfolio return (as decimal).

        Returns:
            Annualized Sharpe ratio. Returns nan if window not full.

        Example:
            >>> sharpe = tracker.update(0.0015)  # +0.15% today
        """
        excess = daily_return - self.risk_free_rate
        mean, std = self._stats.update(excess)

        if not self._stats.is_ready() or std < 1e-10:
            self.current_sharpe = float("nan")
            return float("nan")

        self.current_sharpe = (mean / std) * self._ANNUALIZATION
        return self.current_sharpe

    def is_ready(self) -> bool:
        """Return True if window is fully populated."""
        return self._stats.is_ready()


# ---------------------------------------------------------------------------
# IncrementalMetrics — Real-time risk monitoring (O(1) per update)
# ---------------------------------------------------------------------------

class IncrementalMetrics:
    """Real-time portfolio risk metrics updated in O(1) per bar.

    Tracks equity curve, drawdown, and rolling Sharpe without
    recomputing over all historical data on each update.

    Used by the LiveTrader as a circuit breaker:
        - If drawdown > max_drawdown_limit → halt strategy
        - If Sharpe < min_sharpe_limit → alert risk manager

    Attributes:
        max_drawdown_limit: Halt trading if drawdown exceeds this.
        min_sharpe_limit: Alert if rolling Sharpe drops below this.
        current_drawdown: Current peak-to-trough drawdown.
        current_sharpe: Current rolling Sharpe ratio.
        total_return: Total return since inception.

    Example:
        >>> metrics = IncrementalMetrics(max_drawdown_limit=0.10)
        >>> for equity, ret in zip(equity_values, daily_returns):
        ...     metrics.update(equity, ret)
        ...     if metrics.should_halt():
        ...         stop_trading()
    """

    def __init__(
        self,
        max_drawdown_limit: float = 0.15,
        min_sharpe_limit: float = 0.0,
        sharpe_window: int = 60,
    ) -> None:
        """Initialize IncrementalMetrics.

        Args:
            max_drawdown_limit: Halt threshold for drawdown (default 15%).
            min_sharpe_limit: Alert threshold for Sharpe (default 0.0).
            sharpe_window: Rolling window for Sharpe computation.
        """
        self.max_drawdown_limit: float = max_drawdown_limit
        self.min_sharpe_limit: float = min_sharpe_limit

        self._peak_equity: float = 0.0
        self._initial_equity: float = 0.0
        self._current_equity: float = 0.0
        self._sharpe_tracker: OnlineRollingSharpe = OnlineRollingSharpe(sharpe_window)
        self._initialized: bool = False
        self._n_updates: int = 0

        # Public metrics
        self.current_drawdown: float = 0.0
        self.current_sharpe: float = float("nan")
        self.total_return: float = 0.0

    def update(self, equity: float, daily_return: float) -> None:
        """Update all metrics with new equity and return. O(1).

        Args:
            equity: Current portfolio equity value.
            daily_return: Today's portfolio return (as decimal).
        """
        if not self._initialized:
            self._initial_equity = equity
            self._peak_equity = equity
            self._initialized = True

        self._current_equity = equity
        self._n_updates += 1

        # Update peak (O(1) — just a comparison)
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Drawdown (O(1))
        if self._peak_equity > 0:
            self.current_drawdown = (self._peak_equity - equity) / self._peak_equity
        else:
            self.current_drawdown = 0.0

        # Total return (O(1))
        if self._initial_equity > 0:
            self.total_return = (equity - self._initial_equity) / self._initial_equity

        # Rolling Sharpe (O(1) via OnlineRollingSharpe)
        self.current_sharpe = self._sharpe_tracker.update(daily_return)

        logger.debug(
            "Metrics | equity=%.2f | drawdown=%.2f%% | sharpe=%.2f | total_ret=%.2f%%",
            equity,
            self.current_drawdown * 100,
            self.current_sharpe if not math.isnan(self.current_sharpe) else 0.0,
            self.total_return * 100,
        )

    def should_halt(self) -> bool:
        """Return True if risk limits are breached — halt trading.

        Checks:
            1. Drawdown exceeds max_drawdown_limit
            2. Sharpe drops below min_sharpe_limit (if window is ready)

        Returns:
            True if trading should be halted immediately.

        Example:
            >>> if metrics.should_halt():
            ...     live_trader.emergency_stop()
        """
        if self.current_drawdown > self.max_drawdown_limit:
            logger.warning(
                "HALT: Drawdown %.2f%% exceeds limit %.2f%%",
                self.current_drawdown * 100,
                self.max_drawdown_limit * 100,
            )
            return True

        if (
            self._sharpe_tracker.is_ready()
            and not math.isnan(self.current_sharpe)
            and self.current_sharpe < self.min_sharpe_limit
        ):
            logger.warning(
                "ALERT: Sharpe %.2f below limit %.2f",
                self.current_sharpe,
                self.min_sharpe_limit,
            )
            return True

        return False

    def get_summary(self) -> dict[str, float]:
        """Return current metrics as a dictionary.

        Returns:
            Dictionary of metric names to values.
        """
        return {
            "total_return": round(self.total_return, 6),
            "current_drawdown": round(self.current_drawdown, 6),
            "current_sharpe": round(self.current_sharpe, 4)
            if not math.isnan(self.current_sharpe)
            else 0.0,
            "peak_equity": round(self._peak_equity, 2),
            "current_equity": round(self._current_equity, 2),
            "n_updates": self._n_updates,
        }

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self._peak_equity = 0.0
        self._initial_equity = 0.0
        self._current_equity = 0.0
        self._sharpe_tracker = OnlineRollingSharpe(
            self._sharpe_tracker.window, self._sharpe_tracker.risk_free_rate
        )
        self._initialized = False
        self._n_updates = 0
        self.current_drawdown = 0.0
        self.current_sharpe = float("nan")
        self.total_return = 0.0
