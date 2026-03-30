"""Mean Reversion Trading Strategy — Backtesting + Live Trading modes.

Backtesting mode:  generate_signals(df)       → processes all historical bars
Live trading mode: get_latest_signal(price)   → O(1) per tick via OnlineZScore

Strategy Logic (Bollinger Bands / Z-Score):
    - BUY  when price drops below mean - (z_threshold * std)  → oversold
    - SELL when price rises above mean + (z_threshold * std)  → overbought
    - HOLD when price is within the band

Live trading uses OnlineZScore (Welford's algorithm) which updates in O(1)
per price tick. No historical data recomputation — critical for low latency.

References:
    - Bollinger (1992): "Bollinger on Bollinger Bands"
    - Gatev, Goetzmann & Rouwenhorst (2006): "Pairs Trading"
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from engine.online_indicators import OnlineZScore
from engine.sliding_window import rolling_mean, rolling_std
from engine.strategies.base_strategy import BaseStrategy, Signal, StrategyConfig

logger = get_logger(__name__)

# Default hyperparameters
DEFAULT_WINDOW = 20          # Rolling window for mean/std computation
DEFAULT_Z_THRESHOLD = 2.0    # Number of std deviations to trigger signal
DEFAULT_EXIT_THRESHOLD = 0.0 # Z-score at which to exit (0 = at the mean)


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band / Z-Score Mean Reversion Strategy.

    Supports two execution modes:
        Backtesting:   generate_signals(df)       → batch processing
        Live trading:  get_latest_signal(price)   → O(1) via OnlineZScore

    Attributes:
        window: Rolling window for computing mean and std.
        z_threshold: Number of standard deviations for entry signal.
        exit_threshold: Z-score at which to exit a position.

    Example (backtesting):
        >>> config = StrategyConfig(params={"window": 20, "z_threshold": 2.0})
        >>> strategy = MeanReversionStrategy(config)
        >>> signals = strategy.generate_signals(df)

    Example (live trading):
        >>> strategy = MeanReversionStrategy()
        >>> for price in live_price_feed:
        ...     signal = strategy.get_latest_signal(price)
        ...     if signal == Signal.BUY:
        ...         broker.buy(qty=100)
    """

    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        """Initialize MeanReversionStrategy with optional config.

        Resolves params BEFORE super().__init__() so get_name() and
        validate_params() can access instance attributes safely.

        Args:
            config: StrategyConfig with optional params:
                - window (int): Rolling window size. Default 20.
                - z_threshold (float): Entry z-score threshold. Default 2.0.
                - exit_threshold (float): Exit z-score threshold. Default 0.0.
        """
        resolved_config = config or StrategyConfig()
        params = resolved_config.params
        self.window: int = int(params.get("window", DEFAULT_WINDOW))
        self.z_threshold: float = float(params.get("z_threshold", DEFAULT_Z_THRESHOLD))
        self.exit_threshold: float = float(
            params.get("exit_threshold", DEFAULT_EXIT_THRESHOLD)
        )

        # Online z-score tracker for live trading (O(1) per tick)
        self._zscore_tracker: OnlineZScore = OnlineZScore(
            window=self.window, threshold=self.z_threshold
        )
        # Track current position for exit logic in live mode
        self._live_position: int = Signal.HOLD.value

        super().__init__(resolved_config)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        """Return strategy name including hyperparameters.

        Returns:
            Strategy name string.
        """
        return f"MeanReversion_W{self.window}_Z{self.z_threshold}"

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def validate_params(self) -> None:
        """Validate mean reversion specific parameters.

        Raises:
            ValueError: If window < 2 or z_threshold <= 0.
        """
        if self.window < 2:
            raise ValueError(f"window must be >= 2, got {self.window}.")
        if self.z_threshold <= 0:
            raise ValueError(f"z_threshold must be > 0, got {self.z_threshold}.")

    # ------------------------------------------------------------------
    # Live trading — O(1) per tick
    # ------------------------------------------------------------------

    def get_latest_signal(self, price: float) -> Signal:
        """Generate ONE signal from the latest price tick. O(1).

        Updates the OnlineZScore tracker with the new price and checks
        for entry/exit conditions. Runs in ~1.5 μs.

        Position management:
            - Entry: BUY when z < -threshold, SELL when z > +threshold
            - Exit:  SELL long when z >= exit_threshold
                     BUY short when z <= exit_threshold

        Args:
            price: Latest closing price.

        Returns:
            Signal.BUY, Signal.SELL, or Signal.HOLD.

        Example:
            >>> signal = strategy.get_latest_signal(98.50)
            >>> print(signal)  # Signal.BUY (price is oversold)
        """
        z = self._zscore_tracker.update(price)

        # Window not yet full — no reliable signal
        if not self._zscore_tracker.is_ready():
            return Signal.HOLD

        if self._live_position == Signal.HOLD.value:
            # Entry signals
            if z < -self.z_threshold:
                self._live_position = Signal.BUY.value
                logger.debug(
                    "MR BUY entry | z=%.3f | threshold=%.1f | price=%.2f",
                    z, self.z_threshold, price,
                )
                return Signal.BUY

            if z > self.z_threshold:
                self._live_position = Signal.SELL.value
                logger.debug(
                    "MR SELL entry | z=%.3f | threshold=%.1f | price=%.2f",
                    z, self.z_threshold, price,
                )
                return Signal.SELL

        elif self._live_position == Signal.BUY.value:
            # Exit long when price reverts to mean
            if z >= self.exit_threshold:
                self._live_position = Signal.HOLD.value
                logger.debug(
                    "MR SELL exit (long) | z=%.3f | price=%.2f", z, price
                )
                return Signal.SELL

        elif self._live_position == Signal.SELL.value:
            # Exit short when price reverts to mean
            if z <= self.exit_threshold:
                self._live_position = Signal.HOLD.value
                logger.debug(
                    "MR BUY exit (short) | z=%.3f | price=%.2f", z, price
                )
                return Signal.BUY

        return Signal.HOLD

    def reset_online_state(self) -> None:
        """Reset online z-score tracker and position. Call at session start."""
        super().reset_online_state()
        self._zscore_tracker.reset()
        self._live_position = Signal.HOLD.value
        logger.info("MeanReversionStrategy online state reset | %s", self.get_name())

    # ------------------------------------------------------------------
    # Backtesting — batch processing
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        """Generate BUY/SELL/HOLD signals using Bollinger Band z-score.

        Computes the z-score of the current price relative to its
        rolling mean and standard deviation:

            z_score = (price - rolling_mean) / rolling_std

        Signal logic:
            - BUY  (1):  z_score < -z_threshold  (price is oversold)
            - SELL (-1): z_score > +z_threshold  (price is overbought)
            - HOLD (0):  |z_score| <= z_threshold

        Args:
            df: OHLCV DataFrame with columns ['open','high','low','close','volume'].
                Must have at least window + 1 rows.

        Returns:
            NumPy array of Signal values (1, 0, -1), length = len(df).

        Raises:
            ValueError: If df is missing required columns or is too short.
        """
        self._validate_dataframe(df)

        closes = df["close"].values.astype(float)
        n = len(closes)

        if n < self.window + 1:
            raise ValueError(
                f"DataFrame has {n} rows but strategy requires at least "
                f"{self.window + 1} rows (window + 1)."
            )

        # Compute rolling statistics
        means = rolling_mean(closes, window=self.window)
        stds = rolling_std(closes, window=self.window, ddof=1)

        # Compute z-scores (NaN where window not yet full)
        z_scores = np.full(n, np.nan)
        valid_mask = ~np.isnan(means) & ~np.isnan(stds) & (stds > 1e-10)
        z_scores[valid_mask] = (
            (closes[valid_mask] - means[valid_mask]) / stds[valid_mask]
        )

        # Generate signals with position state tracking
        signals = np.full(n, Signal.HOLD.value, dtype=int)
        position: int = Signal.HOLD.value

        for i in range(self.window, n):
            z = z_scores[i]
            if np.isnan(z):
                continue

            if position == Signal.HOLD.value:
                if z < -self.z_threshold:
                    signals[i] = Signal.BUY.value
                    position = Signal.BUY.value
                elif z > self.z_threshold:
                    signals[i] = Signal.SELL.value
                    position = Signal.SELL.value

            elif position == Signal.BUY.value:
                if z >= self.exit_threshold:
                    signals[i] = Signal.SELL.value
                    position = Signal.HOLD.value

            elif position == Signal.SELL.value:
                if z <= self.exit_threshold:
                    signals[i] = Signal.BUY.value
                    position = Signal.HOLD.value

        n_buys = int(np.sum(signals == Signal.BUY.value))
        n_sells = int(np.sum(signals == Signal.SELL.value))
        logger.info(
            "%s | signals generated | n=%d | buys=%d | sells=%d",
            self.get_name(), n, n_buys, n_sells,
        )
        return signals

    def compute_bollinger_bands(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Bollinger Bands for visualization.

        Returns the upper band, middle band (SMA), and lower band.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Tuple of (upper_band, middle_band, lower_band) as numpy arrays.

        Example:
            >>> upper, mid, lower = strategy.compute_bollinger_bands(df)
        """
        self._validate_dataframe(df)
        closes = df["close"].values.astype(float)
        means = rolling_mean(closes, window=self.window)
        stds = rolling_std(closes, window=self.window, ddof=1)

        upper = means + self.z_threshold * stds
        lower = means - self.z_threshold * stds
        return upper, means, lower
