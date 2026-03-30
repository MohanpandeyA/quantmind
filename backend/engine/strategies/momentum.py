"""Momentum Trading Strategy — Backtesting + Live Trading modes.

Backtesting mode:  generate_signals(df)       → processes all historical bars
Live trading mode: get_latest_signal(price)   → O(1) per tick via OnlineEMA

Strategy Logic (Dual Moving Average Crossover):
    - BUY  when short-term MA crosses ABOVE long-term MA (golden cross)
    - SELL when short-term MA crosses BELOW long-term MA (death cross)
    - HOLD otherwise

Live trading uses OnlineEMA which updates in O(1) per price tick.
No historical data recomputation — critical for sub-millisecond latency.

References:
    - Jegadeesh & Titman (1993): "Returns to Buying Winners and Selling Losers"
    - Faber (2007): "A Quantitative Approach to Tactical Asset Allocation"
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from engine.online_indicators import OnlineEMA
from engine.sliding_window import exponential_moving_average, rolling_mean
from engine.strategies.base_strategy import BaseStrategy, Signal, StrategyConfig

logger = get_logger(__name__)

# Default hyperparameters
DEFAULT_SHORT_WINDOW = 20   # 20-day SMA (1 trading month)
DEFAULT_LONG_WINDOW = 50    # 50-day SMA (2.5 trading months)
DEFAULT_USE_EMA = False     # Use SMA by default; EMA is more responsive


class MomentumStrategy(BaseStrategy):
    """Dual Moving Average Crossover Momentum Strategy.

    Supports two execution modes:
        Backtesting:   generate_signals(df)       → batch processing
        Live trading:  get_latest_signal(price)   → O(1) via OnlineEMA

    Attributes:
        short_window: Look-back period for the fast moving average.
        long_window: Look-back period for the slow moving average.
        use_ema: If True, use EMA instead of SMA.

    Example (backtesting):
        >>> config = StrategyConfig(params={"short_window": 10, "long_window": 30})
        >>> strategy = MomentumStrategy(config)
        >>> signals = strategy.generate_signals(df)

    Example (live trading):
        >>> strategy = MomentumStrategy()
        >>> for price in live_price_feed:
        ...     signal = strategy.get_latest_signal(price)
        ...     if signal == Signal.BUY:
        ...         broker.buy(qty=100)
    """

    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        """Initialize MomentumStrategy with optional config.

        Resolves params BEFORE super().__init__() so get_name() and
        validate_params() can access instance attributes safely.

        Args:
            config: StrategyConfig with optional params:
                - short_window (int): Fast MA period. Default 20.
                - long_window (int): Slow MA period. Default 50.
                - use_ema (bool as 0/1): Use EMA instead of SMA. Default 0.
        """
        resolved_config = config or StrategyConfig()
        params = resolved_config.params
        self.short_window: int = int(params.get("short_window", DEFAULT_SHORT_WINDOW))
        self.long_window: int = int(params.get("long_window", DEFAULT_LONG_WINDOW))
        self.use_ema: bool = bool(params.get("use_ema", DEFAULT_USE_EMA))

        # Online indicators for live trading (O(1) per tick)
        self._short_ema: OnlineEMA = OnlineEMA(span=self.short_window)
        self._long_ema: OnlineEMA = OnlineEMA(span=self.long_window)

        super().__init__(resolved_config)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        """Return strategy name including hyperparameters.

        Returns:
            Strategy name string.
        """
        ma_type = "EMA" if self.use_ema else "SMA"
        return f"Momentum_{ma_type}_{self.short_window}_{self.long_window}"

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def validate_params(self) -> None:
        """Validate momentum-specific parameters.

        Raises:
            ValueError: If short_window >= long_window or windows are invalid.
        """
        if self.short_window < 2:
            raise ValueError(f"short_window must be >= 2, got {self.short_window}.")
        if self.long_window < 2:
            raise ValueError(f"long_window must be >= 2, got {self.long_window}.")
        if self.short_window >= self.long_window:
            raise ValueError(
                f"short_window ({self.short_window}) must be < "
                f"long_window ({self.long_window})."
            )

    # ------------------------------------------------------------------
    # Live trading — O(1) per tick
    # ------------------------------------------------------------------

    def get_latest_signal(self, price: float) -> Signal:
        """Generate ONE signal from the latest price tick. O(1).

        Updates both OnlineEMA trackers with the new price and checks
        for a crossover. No historical data needed — runs in ~1 μs.

        Args:
            price: Latest closing price.

        Returns:
            Signal.BUY on golden cross, Signal.SELL on death cross,
            Signal.HOLD otherwise.

        Example:
            >>> signal = strategy.get_latest_signal(152.30)
            >>> print(signal)  # Signal.HOLD
        """
        prev_short = self._short_ema.current
        prev_long = self._long_ema.current

        # Update both EMAs with new price (O(1) each)
        curr_short = self._short_ema.update(price)
        curr_long = self._long_ema.update(price)

        # Need at least 2 updates before crossover detection
        if not (self._short_ema.count > 1 and self._long_ema.count > 1):
            return Signal.HOLD

        import math
        if math.isnan(prev_short) or math.isnan(prev_long):
            return Signal.HOLD

        # Golden cross: short crosses above long → BUY
        if prev_short <= prev_long and curr_short > curr_long:
            logger.debug(
                "Golden cross | short=%.4f | long=%.4f | price=%.2f",
                curr_short, curr_long, price,
            )
            return Signal.BUY

        # Death cross: short crosses below long → SELL
        if prev_short >= prev_long and curr_short < curr_long:
            logger.debug(
                "Death cross | short=%.4f | long=%.4f | price=%.2f",
                curr_short, curr_long, price,
            )
            return Signal.SELL

        return Signal.HOLD

    def reset_online_state(self) -> None:
        """Reset online EMA trackers. Call at start of each trading session."""
        super().reset_online_state()
        self._short_ema.reset()
        self._long_ema.reset()
        logger.info("MomentumStrategy online state reset | %s", self.get_name())

    # ------------------------------------------------------------------
    # Backtesting — batch processing
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        """Generate BUY/SELL/HOLD signals using moving average crossover.

        Signal logic:
            - BUY  (1):  short_ma[t-1] <= long_ma[t-1] AND short_ma[t] > long_ma[t]
            - SELL (-1): short_ma[t-1] >= long_ma[t-1] AND short_ma[t] < long_ma[t]
            - HOLD (0):  no crossover detected

        Args:
            df: OHLCV DataFrame with columns ['open','high','low','close','volume'].
                Must have at least long_window + 1 rows.

        Returns:
            NumPy array of Signal values (1, 0, -1), length = len(df).

        Raises:
            ValueError: If df is missing required columns or is too short.
        """
        self._validate_dataframe(df)

        closes = df["close"].values.astype(float)
        n = len(closes)

        if n < self.long_window + 1:
            raise ValueError(
                f"DataFrame has {n} rows but strategy requires at least "
                f"{self.long_window + 1} rows (long_window + 1)."
            )

        # Compute moving averages
        if self.use_ema:
            short_ma = exponential_moving_average(closes, span=self.short_window)
            long_ma = exponential_moving_average(closes, span=self.long_window)
        else:
            short_ma = rolling_mean(closes, window=self.short_window)
            long_ma = rolling_mean(closes, window=self.long_window)

        # Initialize all signals as HOLD
        signals = np.full(n, Signal.HOLD.value, dtype=int)

        # Detect crossovers starting from long_window (first valid long_ma)
        for i in range(self.long_window, n):
            prev_short = short_ma[i - 1]
            prev_long = long_ma[i - 1]
            curr_short = short_ma[i]
            curr_long = long_ma[i]

            if np.isnan(prev_short) or np.isnan(prev_long):
                continue
            if np.isnan(curr_short) or np.isnan(curr_long):
                continue

            # Golden cross → BUY
            if prev_short <= prev_long and curr_short > curr_long:
                signals[i] = Signal.BUY.value

            # Death cross → SELL
            elif prev_short >= prev_long and curr_short < curr_long:
                signals[i] = Signal.SELL.value

        n_buys = int(np.sum(signals == Signal.BUY.value))
        n_sells = int(np.sum(signals == Signal.SELL.value))
        logger.info(
            "%s | signals generated | n=%d | buys=%d | sells=%d",
            self.get_name(), n, n_buys, n_sells,
        )
        return signals
