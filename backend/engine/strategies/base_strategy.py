"""Abstract base class for all trading strategies in QuantMind.

Two execution modes:
    Backtesting:   generate_signals(df)       → all historical bars at once
    Live trading:  get_latest_signal(price)   → one tick at a time, O(1)

Every strategy must inherit from BaseStrategy and implement:
- generate_signals(): produce BUY/SELL/HOLD signals from OHLCV data (backtesting)
- get_latest_signal(): produce ONE signal from the latest price tick (live trading)
- get_name(): return a human-readable strategy name

This enforces a consistent interface so the Backtester and LiveTrader can run
any strategy without knowing its internals (Open/Closed Principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger

logger = get_logger(__name__)


class Signal(int, Enum):
    """Trading signal values.

    Attributes:
        BUY:  Enter a long position (value = 1).
        HOLD: Maintain current position (value = 0).
        SELL: Exit long / enter short (value = -1).
    """

    BUY = 1
    HOLD = 0
    SELL = -1


@dataclass
class StrategyConfig:
    """Configuration parameters for a trading strategy.

    Attributes:
        initial_capital: Starting portfolio value in USD.
        position_size: Fraction of capital to deploy per trade (0.0-1.0).
        stop_loss: Maximum loss per trade as a fraction (e.g., 0.02 = 2%).
        take_profit: Target gain per trade as a fraction (e.g., 0.05 = 5%).
        commission: Per-trade commission as a fraction (e.g., 0.001 = 0.1%).
        params: Strategy-specific hyperparameters (e.g., window sizes).
    """

    initial_capital: float = 100_000.0
    position_size: float = 1.0
    stop_loss: float = 0.02
    take_profit: float = 0.05
    commission: float = 0.001
    params: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if not (0 < self.position_size <= 1.0):
            raise ValueError(
                f"position_size must be in (0, 1], got {self.position_size}."
            )
        if self.initial_capital <= 0:
            raise ValueError(
                f"initial_capital must be positive, got {self.initial_capital}."
            )
        if not (0 <= self.stop_loss < 1.0):
            raise ValueError(
                f"stop_loss must be in [0, 1), got {self.stop_loss}."
            )
        if not (0 <= self.commission < 0.1):
            raise ValueError(
                f"commission must be in [0, 0.1), got {self.commission}."
            )


@dataclass
class BacktestResult:
    """Results from a single strategy backtest run.

    Attributes:
        ticker: Asset ticker symbol.
        strategy_name: Name of the strategy used.
        signals: Array of Signal values aligned with price data.
        equity_curve: Portfolio value at each time step.
        returns: Daily portfolio returns.
        trade_returns: Return for each completed trade.
        n_trades: Total number of trades executed.
        config: Strategy configuration used.
    """

    ticker: str
    strategy_name: str
    signals: np.ndarray
    equity_curve: np.ndarray
    returns: np.ndarray
    trade_returns: np.ndarray
    n_trades: int
    config: StrategyConfig


class BaseStrategy(ABC):
    """Abstract base class for all QuantMind trading strategies.

    Two execution modes:
        Backtesting:   generate_signals(df)       → all historical bars at once
        Live trading:  get_latest_signal(price)   → one tick at a time, O(1)

    Subclasses must implement:
        - generate_signals(): batch signal logic for backtesting
        - get_latest_signal(): online signal logic for live trading (O(1))
        - get_name(): strategy identifier

    Subclasses may override:
        - validate_params(): custom parameter validation
        - reset_online_state(): reset online indicators between sessions

    Example:
        >>> class MyStrategy(BaseStrategy):
        ...     def get_name(self) -> str:
        ...         return "MyStrategy"
        ...     def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        ...         return signals
        ...     def get_latest_signal(self, price: float) -> Signal:
        ...         return Signal.HOLD
    """

    def __init__(self, config: Optional[StrategyConfig] = None) -> None:
        """Initialize the strategy with optional configuration.

        Args:
            config: StrategyConfig instance. Uses defaults if None.
        """
        self.config: StrategyConfig = config or StrategyConfig()
        self._df_validated: bool = False  # Validate DataFrame only once (not per bar)
        self.validate_params()
        logger.debug(
            "Strategy initialized | name=%s | capital=%.2f",
            self.get_name(),
            self.config.initial_capital,
        )

    # ------------------------------------------------------------------
    # Abstract methods — MUST implement in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def get_name(self) -> str:
        """Return the human-readable name of this strategy.

        Returns:
            Strategy name string (e.g., "Momentum_SMA_20_50").
        """
        ...

    @abstractmethod
    def get_latest_signal(self, price: float) -> Signal:
        """Generate ONE signal from the latest price tick. O(1).

        This is the LIVE TRADING hot path. Must be as fast as possible.
        Uses online indicators (OnlineEMA, OnlineZScore) that update
        incrementally — no historical data recomputation.

        Args:
            price: Latest closing price (or mid-price for live data).

        Returns:
            Signal enum value: BUY, SELL, or HOLD.

        Example:
            >>> signal = strategy.get_latest_signal(150.25)
            >>> if signal == Signal.BUY:
            ...     broker.place_order(side="buy", qty=100)
        """
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        """Generate BUY/SELL/HOLD signals from OHLCV price data.

        This is the BACKTESTING path. Processes all historical bars at once.

        Args:
            df: DataFrame with columns: ['open', 'high', 'low', 'close', 'volume'].
                Index must be a DatetimeIndex sorted ascending.

        Returns:
            NumPy array of Signal values (1=BUY, 0=HOLD, -1=SELL),
            same length as df.

        Raises:
            ValueError: If required columns are missing from df.
        """
        ...

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def validate_params(self) -> None:
        """Validate strategy-specific parameters.

        Override in subclasses to add custom parameter validation.
        Called automatically during __init__.
        """
        pass

    def reset_online_state(self) -> None:
        """Reset all online indicators to initial state.

        Call this at the start of each trading session to clear
        stale state from the previous session.

        Override in subclasses to reset strategy-specific indicators.
        """
        self._df_validated = False
        logger.info("Online state reset | strategy=%s", self.get_name())

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that the input DataFrame has required OHLCV columns.

        Validation is skipped after the first successful call (cached via
        _df_validated flag) to avoid overhead in the backtesting loop.

        Args:
            df: Input DataFrame to validate.

        Raises:
            ValueError: If required columns are missing or df is empty.
        """
        if self._df_validated:
            return  # Already validated — skip for performance

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns.str.lower())
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}. "
                f"Got: {list(df.columns)}"
            )
        if df.empty:
            raise ValueError("DataFrame must not be empty.")
        if len(df) < 2:
            raise ValueError(
                f"DataFrame must have at least 2 rows, got {len(df)}."
            )

        self._df_validated = True

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.get_name()!r}, "
            f"capital={self.config.initial_capital:.0f}, "
            f"position_size={self.config.position_size:.2f})"
        )
