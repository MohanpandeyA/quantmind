"""RSI (Relative Strength Index) Trading Strategy — Backtesting + Live Trading.

Backtesting mode:  generate_signals(df)       → processes all historical bars
Live trading mode: get_latest_signal(price)   → O(1) per tick via online RSI

Strategy Logic (Wilder's RSI):
    - RSI < oversold  (default 30) → BUY  (price is oversold, likely to bounce)
    - RSI > overbought (default 70) → SELL (price is overbought, likely to fall)
    - 30 ≤ RSI ≤ 70               → HOLD

RSI Formula:
    RS = Average Gain / Average Loss  (over `period` days)
    RSI = 100 - (100 / (1 + RS))

Wilder's Smoothing (used in live mode):
    avg_gain = (prev_avg_gain × (period-1) + current_gain) / period
    avg_loss = (prev_avg_loss × (period-1) + current_loss) / period
    This is O(1) per tick — no need to store all past prices.

When to use RSI vs other strategies:
    - RSI: Best for oscillating/sideways markets with clear overbought/oversold levels
    - Momentum (EMA crossover): Best for smooth trending markets
    - Mean Reversion (Z-score): Best for high-volatility oscillating markets
    - MACD: Best for trending markets with momentum confirmation

References:
    - Wilder (1978): "New Concepts in Technical Trading Systems"
    - Appel (2005): "Technical Analysis: Power Tools for Active Investors"
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from engine.strategies.base_strategy import BaseStrategy, Signal, StrategyConfig

logger = get_logger(__name__)

# Default hyperparameters (industry standard values)
DEFAULT_PERIOD = 14       # Wilder's original 14-day RSI
DEFAULT_OVERSOLD = 30     # Below 30 = oversold → BUY signal
DEFAULT_OVERBOUGHT = 70   # Above 70 = overbought → SELL signal


class RSIStrategy(BaseStrategy):
    """RSI-based mean reversion strategy using Wilder's smoothing.

    Supports two execution modes:
        Backtesting:   generate_signals(df)       → batch processing
        Live trading:  get_latest_signal(price)   → O(1) via online RSI

    Attributes:
        period: RSI look-back period (default 14).
        oversold: RSI threshold below which to BUY (default 30).
        overbought: RSI threshold above which to SELL (default 70).

    Example (backtesting):
        >>> config = StrategyConfig(params={"period": 14, "oversold": 30, "overbought": 70})
        >>> strategy = RSIStrategy(config)
        >>> signals = strategy.generate_signals(df)

    Example (live trading):
        >>> strategy = RSIStrategy()
        >>> for price in live_price_feed:
        ...     signal = strategy.get_latest_signal(price)
        ...     if signal == Signal.BUY:
        ...         broker.buy(qty=100)
    """

    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        """Initialize RSIStrategy with optional config.

        Args:
            config: StrategyConfig with optional params:
                - period (int): RSI look-back period. Default 14.
                - oversold (float): RSI below this → BUY. Default 30.
                - overbought (float): RSI above this → SELL. Default 70.
        """
        resolved_config = config or StrategyConfig()
        params = resolved_config.params
        self.period: int = int(params.get("period", DEFAULT_PERIOD))
        self.oversold: float = float(params.get("oversold", DEFAULT_OVERSOLD))
        self.overbought: float = float(params.get("overbought", DEFAULT_OVERBOUGHT))

        # Online state for live trading (Wilder's smoothing)
        self._prev_price: Optional[float] = None
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._n_ticks: int = 0
        self._warmup_gains: list[float] = []
        self._warmup_losses: list[float] = []

        super().__init__(resolved_config)

    def get_name(self) -> str:
        """Return strategy name with parameters."""
        return f"RSI_P{self.period}_OS{int(self.oversold)}_OB{int(self.overbought)}"

    def validate_params(self) -> None:
        """Validate RSI parameters."""
        if self.period < 2:
            raise ValueError(f"RSI period must be >= 2, got {self.period}")
        if not (0 < self.oversold < self.overbought < 100):
            raise ValueError(
                f"Must have 0 < oversold ({self.oversold}) < "
                f"overbought ({self.overbought}) < 100"
            )

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        """Generate RSI signals for all historical bars.

        Uses vectorized computation for backtesting efficiency.
        Wilder's smoothing applied after the initial `period` warmup.

        Args:
            df: OHLCV DataFrame with 'close' column.

        Returns:
            np.ndarray of shape (n,) with values:
                +1 = BUY  (RSI crossed below oversold)
                -1 = SELL (RSI crossed above overbought)
                 0 = HOLD
        """
        self._validate_dataframe(df)
        closes = df["close"].values.astype(float)
        n = len(closes)
        signals = np.zeros(n, dtype=int)

        if n < self.period + 1:
            logger.warning(
                "%s | insufficient data for RSI | n=%d < period+1=%d",
                self.get_name(), n, self.period + 1,
            )
            return signals

        # Compute price changes
        deltas = np.diff(closes)  # shape (n-1,)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Initial average gain/loss (simple average over first `period` bars)
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])

        # Compute RSI for each bar after warmup using Wilder's smoothing
        rsi_values = np.full(n, np.nan)

        for i in range(self.period, n):
            if avg_loss == 0.0:
                rsi_values[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))

            # Wilder's smoothing for next bar
            if i < n - 1:
                delta_idx = i  # deltas[i] = closes[i+1] - closes[i]
                gain = gains[delta_idx] if delta_idx < len(gains) else 0.0
                loss = losses[delta_idx] if delta_idx < len(losses) else 0.0
                avg_gain = (avg_gain * (self.period - 1) + gain) / self.period
                avg_loss = (avg_loss * (self.period - 1) + loss) / self.period

        # Generate signals on RSI threshold crossings
        # Signal at bar i means: "act at the OPEN of bar i+1"
        for i in range(self.period, n - 1):
            if np.isnan(rsi_values[i]):
                continue
            if rsi_values[i] < self.oversold:
                signals[i + 1] = 1   # BUY next bar
            elif rsi_values[i] > self.overbought:
                signals[i + 1] = -1  # SELL next bar

        buys = int(np.sum(signals == 1))
        sells = int(np.sum(signals == -1))
        logger.info(
            "%s | signals generated | n=%d | buys=%d | sells=%d",
            self.get_name(), n, buys, sells,
        )
        return signals

    def get_latest_signal(self, price: float) -> Signal:
        """O(1) online RSI signal for live trading.

        Uses Wilder's smoothing to maintain running avg_gain/avg_loss
        without storing all past prices.

        Args:
            price: Latest closing price.

        Returns:
            Signal.BUY, Signal.SELL, or Signal.HOLD.
        """
        if self._prev_price is None:
            self._prev_price = price
            return Signal.HOLD

        delta = price - self._prev_price
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        self._prev_price = price
        self._n_ticks += 1

        # Warmup phase: collect first `period` deltas
        if self._n_ticks <= self.period:
            self._warmup_gains.append(gain)
            self._warmup_losses.append(loss)
            if self._n_ticks == self.period:
                # Initialize Wilder's averages
                self._avg_gain = float(np.mean(self._warmup_gains))
                self._avg_loss = float(np.mean(self._warmup_losses))
            return Signal.HOLD

        # Wilder's smoothing (O(1))
        self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
        self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

        if self._avg_loss == 0.0:
            rsi = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        if rsi < self.oversold:
            return Signal.BUY
        elif rsi > self.overbought:
            return Signal.SELL
        return Signal.HOLD

    def reset_online_state(self) -> None:
        """Reset live trading state (call between sessions)."""
        self._prev_price = None
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._n_ticks = 0
        self._warmup_gains = []
        self._warmup_losses = []
