"""MACD (Moving Average Convergence Divergence) Strategy — Backtesting + Live Trading.

Backtesting mode:  generate_signals(df)       → processes all historical bars
Live trading mode: get_latest_signal(price)   → O(1) per tick via online EMA

Strategy Logic (MACD Crossover):
    MACD Line   = EMA(fast) - EMA(slow)       default: EMA(12) - EMA(26)
    Signal Line = EMA(signal_period) of MACD  default: EMA(9) of MACD
    Histogram   = MACD Line - Signal Line

    BUY  when MACD Line crosses ABOVE Signal Line (bullish crossover)
    SELL when MACD Line crosses BELOW Signal Line (bearish crossover)
    HOLD otherwise

Why MACD is better than simple EMA crossover (MomentumStrategy):
    - MomentumStrategy uses SMA/EMA of price directly (slow to react)
    - MACD uses EMA of the DIFFERENCE between two EMAs (faster signal)
    - MACD histogram shows momentum strength, not just direction
    - MACD has built-in noise filtering via the signal line smoothing

When to use MACD vs other strategies:
    - MACD: Best for trending markets with momentum confirmation
    - RSI: Best for oscillating/sideways markets (overbought/oversold)
    - Mean Reversion: Best for high-volatility oscillating markets
    - Momentum (EMA crossover): Simpler alternative for smooth trends

Live trading O(1) implementation:
    EMA update: ema_new = α × price + (1-α) × ema_prev  where α = 2/(period+1)
    This is O(1) per tick — no historical data needed after warmup.

References:
    - Appel (1979): Original MACD paper
    - Murphy (1999): "Technical Analysis of the Financial Markets"
    - Pring (2002): "Technical Analysis Explained"
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from engine.strategies.base_strategy import BaseStrategy, Signal, StrategyConfig

logger = get_logger(__name__)

# Default hyperparameters (industry standard — Appel's original values)
DEFAULT_FAST = 12           # Fast EMA period
DEFAULT_SLOW = 26           # Slow EMA period
DEFAULT_SIGNAL_PERIOD = 9   # Signal line EMA period


class MACDStrategy(BaseStrategy):
    """MACD crossover strategy using triple EMA system.

    Supports two execution modes:
        Backtesting:   generate_signals(df)       → batch processing
        Live trading:  get_latest_signal(price)   → O(1) via online EMA

    Attributes:
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal_period: Signal line EMA period (default 9).

    Example (backtesting):
        >>> config = StrategyConfig(params={"fast": 12, "slow": 26, "signal_period": 9})
        >>> strategy = MACDStrategy(config)
        >>> signals = strategy.generate_signals(df)

    Example (live trading):
        >>> strategy = MACDStrategy()
        >>> for price in live_price_feed:
        ...     signal = strategy.get_latest_signal(price)
        ...     if signal == Signal.BUY:
        ...         broker.buy(qty=100)
    """

    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        """Initialize MACDStrategy with optional config.

        Args:
            config: StrategyConfig with optional params:
                - fast (int): Fast EMA period. Default 12.
                - slow (int): Slow EMA period. Default 26.
                - signal_period (int): Signal line EMA period. Default 9.
        """
        resolved_config = config or StrategyConfig()
        params = resolved_config.params
        self.fast: int = int(params.get("fast", DEFAULT_FAST))
        self.slow: int = int(params.get("slow", DEFAULT_SLOW))
        self.signal_period: int = int(params.get("signal_period", DEFAULT_SIGNAL_PERIOD))

        # EMA smoothing factors
        self._alpha_fast: float = 2.0 / (self.fast + 1)
        self._alpha_slow: float = 2.0 / (self.slow + 1)
        self._alpha_signal: float = 2.0 / (self.signal_period + 1)

        # Online state for live trading
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._macd_signal: Optional[float] = None
        self._prev_macd: Optional[float] = None
        self._prev_signal: Optional[float] = None
        self._n_ticks: int = 0
        self._warmup_prices: list[float] = []

        super().__init__(resolved_config)

    def get_name(self) -> str:
        """Return strategy name with parameters."""
        return f"MACD_{self.fast}_{self.slow}_{self.signal_period}"

    def validate_params(self) -> None:
        """Validate MACD parameters."""
        if self.fast >= self.slow:
            raise ValueError(
                f"MACD fast period ({self.fast}) must be < slow period ({self.slow})"
            )
        if self.fast < 2:
            raise ValueError(f"MACD fast period must be >= 2, got {self.fast}")
        if self.signal_period < 2:
            raise ValueError(f"MACD signal period must be >= 2, got {self.signal_period}")

    @staticmethod
    def _compute_ema(values: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA array using vectorized Wilder-style smoothing.

        Args:
            values: Price array.
            period: EMA period.

        Returns:
            EMA array of same length as values.
        """
        alpha = 2.0 / (period + 1)
        ema = np.empty(len(values))
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1.0 - alpha) * ema[i - 1]
        return ema

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        """Generate MACD crossover signals for all historical bars.

        Computes three EMA arrays (fast, slow, signal) and detects
        crossovers between MACD line and signal line.

        Args:
            df: OHLCV DataFrame with 'close' column.

        Returns:
            np.ndarray of shape (n,) with values:
                +1 = BUY  (MACD crossed above signal line)
                -1 = SELL (MACD crossed below signal line)
                 0 = HOLD
        """
        self._validate_dataframe(df)
        closes = df["close"].values.astype(float)
        n = len(closes)
        signals = np.zeros(n, dtype=int)

        min_bars = self.slow + self.signal_period
        if n < min_bars:
            logger.warning(
                "%s | insufficient data | n=%d < min_bars=%d",
                self.get_name(), n, min_bars,
            )
            return signals

        # Compute fast and slow EMAs
        ema_fast = self._compute_ema(closes, self.fast)
        ema_slow = self._compute_ema(closes, self.slow)

        # MACD line = fast EMA - slow EMA
        macd_line = ema_fast - ema_slow

        # Signal line = EMA of MACD line
        signal_line = self._compute_ema(macd_line, self.signal_period)

        # Detect crossovers: signal at bar i → act at bar i+1
        # Crossover: MACD was below signal, now above (or vice versa)
        for i in range(self.slow + self.signal_period - 1, n - 1):
            prev_diff = macd_line[i - 1] - signal_line[i - 1]
            curr_diff = macd_line[i] - signal_line[i]

            if prev_diff < 0 and curr_diff >= 0:
                # MACD crossed ABOVE signal → bullish → BUY
                signals[i + 1] = 1
            elif prev_diff > 0 and curr_diff <= 0:
                # MACD crossed BELOW signal → bearish → SELL
                signals[i + 1] = -1

        buys = int(np.sum(signals == 1))
        sells = int(np.sum(signals == -1))
        logger.info(
            "%s | signals generated | n=%d | buys=%d | sells=%d",
            self.get_name(), n, buys, sells,
        )
        return signals

    def get_latest_signal(self, price: float) -> Signal:
        """O(1) online MACD signal for live trading.

        Maintains running EMA values using the recurrence relation:
            ema_new = α × price + (1-α) × ema_prev

        Args:
            price: Latest closing price.

        Returns:
            Signal.BUY, Signal.SELL, or Signal.HOLD.
        """
        self._n_ticks += 1

        # Warmup: need at least `slow` prices to initialize EMAs
        if self._n_ticks <= self.slow:
            self._warmup_prices.append(price)
            if self._n_ticks == self.slow:
                # Initialize EMAs with SMA of warmup prices
                prices_arr = np.array(self._warmup_prices)
                self._ema_fast = float(np.mean(prices_arr[-self.fast:]))
                self._ema_slow = float(np.mean(prices_arr))
                macd = self._ema_fast - self._ema_slow
                self._macd_signal = macd
                self._prev_macd = macd
                self._prev_signal = macd
            return Signal.HOLD

        # Update fast and slow EMAs (O(1))
        self._ema_fast = self._alpha_fast * price + (1.0 - self._alpha_fast) * self._ema_fast
        self._ema_slow = self._alpha_slow * price + (1.0 - self._alpha_slow) * self._ema_slow

        # Compute MACD line
        macd = self._ema_fast - self._ema_slow

        # Update signal line (EMA of MACD)
        self._macd_signal = (
            self._alpha_signal * macd + (1.0 - self._alpha_signal) * self._macd_signal
        )

        # Detect crossover
        prev_diff = self._prev_macd - self._prev_signal
        curr_diff = macd - self._macd_signal

        self._prev_macd = macd
        self._prev_signal = self._macd_signal

        if prev_diff < 0 and curr_diff >= 0:
            return Signal.BUY
        elif prev_diff > 0 and curr_diff <= 0:
            return Signal.SELL
        return Signal.HOLD

    def reset_online_state(self) -> None:
        """Reset live trading state (call between sessions)."""
        self._ema_fast = None
        self._ema_slow = None
        self._macd_signal = None
        self._prev_macd = None
        self._prev_signal = None
        self._n_ticks = 0
        self._warmup_prices = []
