"""Core backtesting engine for QuantMind.

SSL fix for macOS LibreSSL — applied at module load time.

The Backtester takes a strategy and OHLCV price data, simulates trading
by following the strategy's signals, and produces a BacktestResult with
a full equity curve, returns series, and trade log.

Key design decisions:
- Uses yfinance for free market data (no API key needed)
- Uses SegmentTree for O(log n) support/resistance queries
- Simulates realistic trading with commission costs
- Supports both long-only and long/short modes
- Returns structured BacktestResult for downstream metrics computation

Simulation assumptions:
- Trades execute at the NEXT day's open price (avoids look-ahead bias)
- Commission is applied on both entry and exit
- No slippage model (can be added as upgrade)
- No fractional shares
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config.logging_config import get_logger
from engine.metrics import PerformanceReport, compute_full_report
from engine.segment_tree import AggregationType, SegmentTree, build_price_trees
from engine.strategies.base_strategy import BacktestResult, BaseStrategy, Signal, StrategyConfig

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run.

    Attributes:
        ticker: Asset ticker symbol (e.g., 'AAPL', 'MSFT').
        start_date: Start date string in 'YYYY-MM-DD' format.
        end_date: End date string in 'YYYY-MM-DD' format.
        long_only: If True, only take long positions (no shorting).
        use_segment_tree: If True, build price trees for range queries.
    """

    ticker: str
    start_date: str
    end_date: str
    long_only: bool = True
    use_segment_tree: bool = True


class Backtester:
    """Simulates trading a strategy on historical OHLCV data.

    Fetches data via yfinance (free, no API key), runs the strategy's
    signal generation, simulates order execution, and computes the
    full equity curve and performance metrics.

    Attributes:
        config: BacktestConfig with ticker and date range.
        strategy: BaseStrategy subclass to backtest.

    Example:
        >>> from engine.strategies.momentum import MomentumStrategy
        >>> strategy = MomentumStrategy()
        >>> bt_config = BacktestConfig("AAPL", "2020-01-01", "2023-12-31")
        >>> backtester = Backtester(bt_config, strategy)
        >>> result, report = backtester.run()
        >>> print(f"Sharpe: {report.sharpe_ratio:.2f}")
    """

    def __init__(self, config: BacktestConfig, strategy: BaseStrategy) -> None:
        """Initialize the Backtester.

        Args:
            config: BacktestConfig specifying ticker and date range.
            strategy: Instantiated strategy to backtest.
        """
        self.config = config
        self.strategy = strategy
        self._df: Optional[pd.DataFrame] = None
        self._max_tree: Optional[SegmentTree] = None
        self._min_tree: Optional[SegmentTree] = None

    def fetch_data(self) -> pd.DataFrame:
        """Download OHLCV data from Yahoo Finance via yfinance.

        Data is cached in self._df after first fetch.

        Returns:
            DataFrame with lowercase columns: open, high, low, close, volume.
            Index is a DatetimeIndex sorted ascending.

        Raises:
            ValueError: If no data is returned for the ticker/date range.

        Example:
            >>> df = backtester.fetch_data()
            >>> df.columns.tolist()
            ['open', 'high', 'low', 'close', 'volume']
        """
        if self._df is not None:
            return self._df

        logger.info(
            "Fetching data | ticker=%s | %s → %s",
            self.config.ticker,
            self.config.start_date,
            self.config.end_date,
        )

        raw = yf.download(
            self.config.ticker,
            start=self.config.start_date,
            end=self.config.end_date,
            progress=False,
            auto_adjust=True,
        )

        if raw.empty:
            raise ValueError(
                f"No data returned for ticker '{self.config.ticker}' "
                f"between {self.config.start_date} and {self.config.end_date}."
            )

        # Normalize column names to lowercase
        # yfinance 1.x returns MultiIndex columns like ('Close', 'AAPL')
        # Flatten to simple lowercase strings
        df = raw.copy()
        if hasattr(df.columns, "levels"):
            # MultiIndex: take the first level (metric name), lowercase it
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        else:
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

        # Ensure required columns exist
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Downloaded data missing columns: {missing}")

        df = df.sort_index()
        self._df = df

        logger.info(
            "Data fetched | ticker=%s | rows=%d | %s → %s",
            self.config.ticker,
            len(df),
            df.index[0].date(),
            df.index[-1].date(),
        )
        return df

    def build_price_trees(self, df: pd.DataFrame) -> None:
        """Build SegmentTree structures for O(log n) range queries.

        Creates a MAX tree on highs and MIN tree on lows.
        Used for support/resistance detection during signal analysis.

        Args:
            df: OHLCV DataFrame.
        """
        highs = df["high"].values.tolist()
        lows = df["low"].values.tolist()
        self._max_tree, self._min_tree = build_price_trees(highs, lows)
        logger.info(
            "Price trees built | ticker=%s | n=%d", self.config.ticker, len(highs)
        )

    def query_resistance(self, left: int, right: int) -> float:
        """Query the highest high (resistance level) in a date range.

        Uses the SegmentTree for O(log n) lookup.

        Args:
            left: Start index (0-based).
            right: End index (0-based, inclusive).

        Returns:
            Highest high price in the range.

        Raises:
            RuntimeError: If price trees have not been built yet.
        """
        if self._max_tree is None:
            raise RuntimeError("Call build_price_trees() before querying.")
        return self._max_tree.query(left, right)

    def query_support(self, left: int, right: int) -> float:
        """Query the lowest low (support level) in a date range.

        Uses the SegmentTree for O(log n) lookup.

        Args:
            left: Start index (0-based).
            right: End index (0-based, inclusive).

        Returns:
            Lowest low price in the range.

        Raises:
            RuntimeError: If price trees have not been built yet.
        """
        if self._min_tree is None:
            raise RuntimeError("Call build_price_trees() before querying.")
        return self._min_tree.query(left, right)

    def run(self) -> tuple[BacktestResult, PerformanceReport]:
        """Execute the full backtest pipeline.

        Steps:
            1. Fetch OHLCV data from Yahoo Finance
            2. Build SegmentTree price structures
            3. Generate signals via strategy
            4. Simulate trade execution (next-day open, with commission)
            5. Compute equity curve and returns
            6. Compute full performance report

        Returns:
            Tuple of (BacktestResult, PerformanceReport).

        Raises:
            ValueError: If data fetch fails or strategy raises an error.

        Example:
            >>> result, report = backtester.run()
            >>> print(report.to_dict())
        """
        # Step 1: Fetch data
        df = self.fetch_data()

        # Step 2: Build price trees
        if self.config.use_segment_tree:
            self.build_price_trees(df)

        # Step 3: Generate signals
        signals = self.strategy.generate_signals(df)

        # Step 4: Simulate execution
        equity_curve, returns, trade_returns = self._simulate_execution(
            df, signals
        )

        # Step 5: Build result
        result = BacktestResult(
            ticker=self.config.ticker,
            strategy_name=self.strategy.get_name(),
            signals=signals,
            equity_curve=equity_curve,
            returns=returns,
            trade_returns=np.array(trade_returns),
            n_trades=len(trade_returns),
            config=self.strategy.config,
        )

        # Step 6: Compute performance report
        report = compute_full_report(
            returns=returns,
            equity_curve=equity_curve,
            trade_returns=np.array(trade_returns),
        )

        logger.info(
            "Backtest complete | %s | %s | sharpe=%.2f | mdd=%.1f%% | trades=%d",
            self.config.ticker,
            self.strategy.get_name(),
            report.sharpe_ratio,
            report.max_drawdown * 100,
            result.n_trades,
        )
        return result, report

    def _simulate_execution(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """Simulate order execution from signals.

        Execution rules:
        - Trades execute at the NEXT bar's open price (avoids look-ahead bias)
        - Commission applied on both entry and exit
        - Long-only mode: only BUY/SELL signals for long positions
        - Position sizing: config.position_size fraction of current equity

        Args:
            df: OHLCV DataFrame.
            signals: Array of Signal values from strategy.

        Returns:
            Tuple of:
                - equity_curve: Array of portfolio values at each bar.
                - returns: Array of daily portfolio returns.
                - trade_returns: List of per-trade returns.
        """
        n = len(df)
        opens = df["open"].values.astype(float)
        closes = df["close"].values.astype(float)

        capital = self.strategy.config.initial_capital
        position_size = self.strategy.config.position_size
        commission = self.strategy.config.commission

        equity_curve = np.zeros(n)
        equity_curve[0] = capital

        trade_returns: list[float] = []
        position: int = 0          # 0 = flat, 1 = long, -1 = short
        entry_price: float = 0.0
        entry_capital: float = 0.0
        shares: float = 0.0

        for i in range(n):
            sig = signals[i]

            # Execute signal at NEXT bar's open (i+1), if available
            exec_idx = i + 1
            if exec_idx >= n:
                # Last bar — mark to market at close price
                if position == 1:
                    # Long position: value = cash + shares * current_price
                    equity_curve[i] = capital + shares * closes[i]
                elif position == -1:
                    # Short position: value = cash + (entry - current) * shares
                    equity_curve[i] = capital + shares * (entry_price - closes[i])
                else:
                    equity_curve[i] = capital
                break

            exec_price = opens[exec_idx]

            # --- Entry ---
            if sig == Signal.BUY.value and position == 0:
                deploy = capital * position_size
                cost = deploy * commission
                shares = (deploy - cost) / exec_price
                entry_price = exec_price
                entry_capital = deploy
                position = 1
                capital -= deploy
                logger.debug(
                    "BUY | bar=%d | price=%.2f | shares=%.4f", i, exec_price, shares
                )

            elif sig == Signal.SELL.value and position == 0 and not self.config.long_only:
                # Short entry (only if not long_only)
                deploy = capital * position_size
                cost = deploy * commission
                shares = (deploy - cost) / exec_price
                entry_price = exec_price
                entry_capital = deploy
                position = -1
                capital -= deploy
                logger.debug(
                    "SHORT | bar=%d | price=%.2f | shares=%.4f", i, exec_price, shares
                )

            # --- Exit ---
            elif sig == Signal.SELL.value and position == 1:
                # Exit long
                proceeds = shares * exec_price
                cost = proceeds * commission
                net_proceeds = proceeds - cost
                trade_ret = (net_proceeds - entry_capital) / entry_capital
                trade_returns.append(trade_ret)
                capital += net_proceeds
                position = 0
                shares = 0.0
                logger.debug(
                    "SELL | bar=%d | price=%.2f | trade_ret=%.4f",
                    i, exec_price, trade_ret,
                )

            elif sig == Signal.BUY.value and position == -1:
                # Cover short
                cost_to_cover = shares * exec_price
                commission_cost = cost_to_cover * commission
                net_cost = cost_to_cover + commission_cost
                trade_ret = (entry_capital - net_cost) / entry_capital
                trade_returns.append(trade_ret)
                capital += entry_capital - net_cost
                position = 0
                shares = 0.0
                logger.debug(
                    "COVER | bar=%d | price=%.2f | trade_ret=%.4f",
                    i, exec_price, trade_ret,
                )

            # Mark to market
            if position == 1:
                equity_curve[i] = capital + shares * closes[i]
            elif position == -1:
                unrealized = entry_price - closes[i]
                equity_curve[i] = capital + shares * unrealized
            else:
                equity_curve[i] = capital

        # Compute daily returns from equity curve
        returns = np.diff(equity_curve) / np.where(
            equity_curve[:-1] > 0, equity_curve[:-1], 1.0
        )

        return equity_curve, returns, trade_returns
