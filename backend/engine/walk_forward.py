"""Walk-Forward Validation for QuantMind backtesting engine.

WHY WALK-FORWARD VALIDATION:
    Standard backtesting tests a strategy on the SAME data used to select it.
    This is like studying the exam answers before the exam — of course it looks good.
    This is called "overfitting" or "in-sample bias".

    Walk-forward validation splits data into rolling train/test windows:
        Train window: optimize strategy parameters (find best params)
        Test window:  test those params on UNSEEN future data

    The test results are stitched together to form the "out-of-sample" equity curve.
    This is what you'd actually get in real trading.

HOW IT WORKS:
    Given 3 years of data (2022-2024) with train=12mo, test=3mo, step=3mo:

    Window 1: Train 2022-01→2022-12, Test 2023-01→2023-03
    Window 2: Train 2022-04→2023-03, Test 2023-04→2023-06
    Window 3: Train 2022-07→2023-06, Test 2023-07→2023-09
    Window 4: Train 2022-10→2023-09, Test 2023-10→2023-12
    Window 5: Train 2023-01→2023-12, Test 2024-01→2024-03
    ...

    For each window:
        1. Grid search on TRAIN data → find best parameters
        2. Apply those parameters to TEST data → get out-of-sample result

ROBUSTNESS RATIO:
    robustness = out_of_sample_sharpe / in_sample_sharpe

    > 0.7  → ROBUST    (strategy holds up on unseen data)
    0.4-0.7 → MODERATE  (some overfitting, use with caution)
    < 0.4  → OVERFITTED (strategy only works on historical data)

REFERENCES:
    - Pardo (2008): "The Evaluation and Optimization of Trading Strategies"
    - Bailey & Lopez de Prado (2014): "The Deflated Sharpe Ratio"
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from engine.backtester import Backtester, BacktestConfig
from engine.metrics import PerformanceReport, compute_full_report
from engine.strategies.base_strategy import StrategyConfig
from engine.strategies.macd_strategy import MACDStrategy
from engine.strategies.mean_reversion import MeanReversionStrategy
from engine.strategies.momentum import MomentumStrategy
from engine.strategies.rsi_strategy import RSIStrategy

logger = get_logger(__name__)

# Parameter grids (same as optimizer)
PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "momentum": {
        "short_window": [5, 10, 20],
        "long_window": [20, 50, 100],
    },
    "mean_reversion": {
        "window": [10, 20, 30],
        "z_threshold": [1.5, 2.0, 2.5],
    },
    "rsi": {
        "period": [7, 14, 21],
        "oversold": [25, 30],
        "overbought": [70, 75],
    },
    "macd": {
        "fast": [8, 12],
        "slow": [21, 26],
        "signal_period": [7, 9],
    },
}

# Robustness thresholds
ROBUST_THRESHOLD = 0.7
MODERATE_THRESHOLD = 0.4


@dataclass
class WindowResult:
    """Result for a single train/test window.

    Attributes:
        window_idx: Window number (1-based).
        train_start: Training period start date.
        train_end: Training period end date.
        test_start: Test period start date.
        test_end: Test period end date.
        best_params: Parameters found by optimizing on train data.
        train_sharpe: Sharpe ratio on training data (in-sample).
        test_sharpe: Sharpe ratio on test data (out-of-sample).
        test_return: Total return on test data.
        test_max_drawdown: Max drawdown on test data.
        test_n_trades: Number of trades in test period.
        test_equity_curve: Equity curve for test period.
    """

    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict[str, Any]
    train_sharpe: float
    test_sharpe: float
    test_return: float
    test_max_drawdown: float
    test_n_trades: int
    test_equity_curve: List[float] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation result.

    Attributes:
        ticker: Stock ticker analyzed.
        strategy: Strategy name.
        in_sample_sharpe: Average Sharpe across all train windows.
        out_of_sample_sharpe: Average Sharpe across all test windows.
        robustness_ratio: out_of_sample / in_sample (higher = less overfitted).
        verdict: ROBUST / MODERATE / OVERFITTED.
        n_windows: Number of walk-forward windows.
        windows: Individual window results.
        combined_equity_curve: Stitched out-of-sample equity curve.
        combined_return: Total return across all test windows.
        combined_max_drawdown: Max drawdown across all test windows.
        processing_time_ms: Total computation time.
    """

    ticker: str
    strategy: str
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    robustness_ratio: float
    verdict: str
    n_windows: int
    windows: List[WindowResult]
    combined_equity_curve: List[float]
    combined_return: float
    combined_max_drawdown: float
    processing_time_ms: float = 0.0


def run_walk_forward(
    ticker: str,
    strategy_name: str,
    start_date: str,
    end_date: str,
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    optimize_for: str = "sharpe",
) -> WalkForwardResult:
    """Run walk-forward validation for a strategy on a ticker.

    Args:
        ticker: Stock ticker symbol.
        strategy_name: Strategy to validate ('momentum', 'mean_reversion', 'rsi', 'macd').
        start_date: Overall start date (YYYY-MM-DD).
        end_date: Overall end date (YYYY-MM-DD).
        train_months: Number of months for each training window.
        test_months: Number of months for each test window.
        step_months: How many months to advance between windows.
        optimize_for: Metric to optimize ('sharpe', 'total_return', 'calmar').

    Returns:
        WalkForwardResult with in-sample vs out-of-sample comparison.
    """
    import time
    start_time = time.perf_counter()

    logger.info(
        "WalkForward | starting | ticker=%s | strategy=%s | "
        "train=%dmo | test=%dmo | step=%dmo",
        ticker, strategy_name, train_months, test_months, step_months,
    )

    # Download data once (reuse across all windows)
    df = _download_data(ticker, start_date, end_date)
    if df is None or len(df) < (train_months + test_months) * 15:
        raise ValueError(
            f"Insufficient data for walk-forward validation. "
            f"Need at least {(train_months + test_months) * 15} trading days."
        )

    # Generate rolling windows
    windows_dates = _generate_windows(start_date, end_date, train_months, test_months, step_months)

    if len(windows_dates) < 2:
        raise ValueError(
            f"Not enough data for walk-forward with train={train_months}mo, "
            f"test={test_months}mo. Try shorter windows or longer date range."
        )

    logger.info("WalkForward | windows=%d | ticker=%s", len(windows_dates), ticker)

    # Run each window
    window_results: List[WindowResult] = []
    for idx, (train_start, train_end, test_start, test_end) in enumerate(windows_dates, 1):
        try:
            result = _run_single_window(
                df=df,
                ticker=ticker,
                strategy_name=strategy_name,
                window_idx=idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                optimize_for=optimize_for,
            )
            window_results.append(result)
            logger.info(
                "WalkForward | window=%d/%d | train_sharpe=%.2f | test_sharpe=%.2f",
                idx, len(windows_dates), result.train_sharpe, result.test_sharpe,
            )
        except Exception as e:
            logger.warning("WalkForward | window=%d failed | %s", idx, e)
            continue

    if not window_results:
        raise ValueError("All walk-forward windows failed.")

    # Aggregate results
    in_sample_sharpe = float(np.mean([w.train_sharpe for w in window_results]))
    out_of_sample_sharpe = float(np.mean([w.test_sharpe for w in window_results]))

    # Robustness ratio (handle edge cases)
    if abs(in_sample_sharpe) < 0.01:
        robustness_ratio = 0.0
    else:
        robustness_ratio = out_of_sample_sharpe / in_sample_sharpe

    # Verdict
    if robustness_ratio >= ROBUST_THRESHOLD:
        verdict = "ROBUST"
    elif robustness_ratio >= MODERATE_THRESHOLD:
        verdict = "MODERATE"
    else:
        verdict = "OVERFITTED"

    # Stitch out-of-sample equity curves
    combined_equity = _stitch_equity_curves(window_results)

    # Combined metrics
    combined_return = (combined_equity[-1] / combined_equity[0] - 1) if len(combined_equity) > 1 else 0.0
    combined_mdd = _compute_max_drawdown(np.array(combined_equity))

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "WalkForward | complete | ticker=%s | in_sample=%.2f | out_of_sample=%.2f | "
        "robustness=%.2f | verdict=%s | time=%.0fms",
        ticker, in_sample_sharpe, out_of_sample_sharpe,
        robustness_ratio, verdict, duration_ms,
    )

    return WalkForwardResult(
        ticker=ticker,
        strategy=strategy_name,
        in_sample_sharpe=round(in_sample_sharpe, 3),
        out_of_sample_sharpe=round(out_of_sample_sharpe, 3),
        robustness_ratio=round(robustness_ratio, 3),
        verdict=verdict,
        n_windows=len(window_results),
        windows=window_results,
        combined_equity_curve=combined_equity,
        combined_return=round(combined_return, 4),
        combined_max_drawdown=round(combined_mdd, 4),
        processing_time_ms=round(duration_ms, 1),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _download_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Download OHLCV data once for all windows.

    Args:
        ticker: Stock ticker.
        start_date: Start date.
        end_date: End date.

    Returns:
        DataFrame with OHLCV data or None on failure.
    """
    warnings.filterwarnings("ignore")
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=start_date, end=end_date,
                          auto_adjust=True, progress=False)
        if raw.empty:
            return None

        # Flatten MultiIndex columns
        if hasattr(raw.columns, "levels"):
            raw.columns = [
                c[0].lower() if isinstance(c, tuple) else c.lower()
                for c in raw.columns
            ]
        else:
            raw.columns = [c.lower() for c in raw.columns]

        raw.index = pd.to_datetime(raw.index)
        return raw
    except Exception as e:
        logger.error("WalkForward | data download failed | %s", e)
        return None


def _generate_windows(
    start_date: str,
    end_date: str,
    train_months: int,
    test_months: int,
    step_months: int,
) -> List[Tuple[str, str, str, str]]:
    """Generate rolling train/test window date ranges.

    Args:
        start_date: Overall start date.
        end_date: Overall end date.
        train_months: Training window length in months.
        test_months: Test window length in months.
        step_months: Step size between windows in months.

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples.
    """
    windows = []
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    current = start
    while True:
        train_start = current
        train_end = _add_months(current, train_months) - timedelta(days=1)
        test_start = _add_months(current, train_months)
        test_end = _add_months(current, train_months + test_months) - timedelta(days=1)

        if test_end > end:
            break

        windows.append((
            train_start.isoformat(),
            train_end.isoformat(),
            test_start.isoformat(),
            test_end.isoformat(),
        ))

        current = _add_months(current, step_months)

    return windows


def _add_months(d: date, months: int) -> date:
    """Add months to a date, handling month-end edge cases.

    Args:
        d: Base date.
        months: Number of months to add.

    Returns:
        New date with months added.
    """
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
                       else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return date(year, month, day)


def _run_single_window(
    df: pd.DataFrame,
    ticker: str,
    strategy_name: str,
    window_idx: int,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    optimize_for: str,
) -> WindowResult:
    """Run one train/test window.

    Args:
        df: Full OHLCV DataFrame.
        ticker: Stock ticker.
        strategy_name: Strategy name.
        window_idx: Window number.
        train_start/end: Training period dates.
        test_start/end: Test period dates.
        optimize_for: Metric to optimize.

    Returns:
        WindowResult with train and test metrics.
    """
    # Slice data for train and test periods
    train_df = _slice_df(df, train_start, train_end)
    test_df = _slice_df(df, test_start, test_end)

    if len(train_df) < 30 or len(test_df) < 5:
        raise ValueError(f"Window {window_idx}: insufficient data")

    # Step 1: Optimize on TRAIN data
    best_params, train_sharpe = _optimize_on_window(
        train_df, strategy_name, optimize_for
    )

    # Step 2: Test best params on TEST data (unseen)
    test_result = _backtest_on_window(test_df, strategy_name, best_params)

    return WindowResult(
        window_idx=window_idx,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        best_params=best_params,
        train_sharpe=round(train_sharpe, 3),
        test_sharpe=round(test_result["sharpe"], 3),
        test_return=round(test_result["total_return"], 4),
        test_max_drawdown=round(test_result["max_drawdown"], 4),
        test_n_trades=test_result["n_trades"],
        test_equity_curve=test_result["equity_curve"],
    )


def _slice_df(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Slice DataFrame to a date range.

    Args:
        df: Full DataFrame with DatetimeIndex.
        start: Start date string.
        end: End date string.

    Returns:
        Sliced DataFrame.
    """
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df[mask].copy()


def _optimize_on_window(
    df: pd.DataFrame,
    strategy_name: str,
    optimize_for: str,
) -> Tuple[Dict[str, Any], float]:
    """Grid search for best parameters on training data.

    Args:
        df: Training DataFrame.
        strategy_name: Strategy name.
        optimize_for: Metric to maximize.

    Returns:
        Tuple of (best_params, best_score).
    """
    import itertools

    grid = PARAM_GRIDS.get(strategy_name, PARAM_GRIDS["momentum"])
    keys = list(grid.keys())
    values = list(grid.values())

    best_score = float("-inf")
    best_params: Dict[str, Any] = {k: v[0] for k, v in grid.items()}

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))

        # Skip invalid combos
        if strategy_name == "momentum":
            if params.get("short_window", 0) >= params.get("long_window", 0):
                continue
        if strategy_name == "macd":
            if params.get("fast", 0) >= params.get("slow", 0):
                continue
        if strategy_name == "rsi":
            if params.get("oversold", 0) >= params.get("overbought", 100):
                continue

        result = _backtest_on_window(df, strategy_name, params)
        score = result.get(optimize_for if optimize_for != "calmar" else "calmar_ratio", 0.0)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score


def _backtest_on_window(
    df: pd.DataFrame,
    strategy_name: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a backtest on a DataFrame slice.

    Args:
        df: OHLCV DataFrame slice.
        strategy_name: Strategy name.
        params: Strategy parameters.

    Returns:
        Dict with sharpe, total_return, max_drawdown, n_trades, equity_curve.
    """
    if len(df) < 10:
        return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                "n_trades": 0, "equity_curve": [100_000.0], "calmar_ratio": 0.0}

    config = StrategyConfig(
        initial_capital=100_000.0,
        position_size=1.0,
        commission=0.001,
        params=params,
    )

    strategy_map = {
        "momentum": MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
        "rsi": RSIStrategy,
        "macd": MACDStrategy,
    }
    StrategyClass = strategy_map.get(strategy_name, MomentumStrategy)
    strategy = StrategyClass(config)

    try:
        signals = strategy.generate_signals(df)
        equity_curve, returns, trade_returns = _simulate(df, signals, config)
        report = compute_full_report(
            returns=returns,
            equity_curve=equity_curve,
            trade_returns=np.array(trade_returns),
        )
        return {
            "sharpe": report.sharpe_ratio,
            "total_return": report.total_return,
            "max_drawdown": report.max_drawdown,
            "n_trades": len(trade_returns),
            "equity_curve": equity_curve.tolist(),
            "calmar_ratio": report.calmar_ratio,
        }
    except Exception:
        return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                "n_trades": 0, "equity_curve": [100_000.0], "calmar_ratio": 0.0}


def _simulate(
    df: pd.DataFrame,
    signals: np.ndarray,
    config: StrategyConfig,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Simplified trade simulation (same logic as Backtester._simulate_execution).

    Args:
        df: OHLCV DataFrame.
        signals: Signal array.
        config: Strategy config.

    Returns:
        Tuple of (equity_curve, returns, trade_returns).
    """
    from engine.strategies.base_strategy import Signal

    n = len(df)
    opens = df["open"].values.astype(float)
    closes = df["close"].values.astype(float)

    capital = config.initial_capital
    commission = config.commission
    equity_curve = np.zeros(n)
    equity_curve[0] = capital
    trade_returns: List[float] = []
    position = 0
    entry_price = 0.0
    entry_capital = 0.0
    shares = 0.0

    for i in range(n):
        sig = signals[i]
        exec_idx = i + 1

        if exec_idx >= n:
            if position == 1:
                equity_curve[i] = capital + shares * closes[i]
            else:
                equity_curve[i] = capital
            break

        exec_price = opens[exec_idx]

        if sig == Signal.BUY.value and position == 0:
            deploy = capital * config.position_size
            cost = deploy * commission
            shares = (deploy - cost) / exec_price
            entry_price = exec_price
            entry_capital = deploy
            position = 1
            capital -= deploy

        elif sig == Signal.SELL.value and position == 1:
            proceeds = shares * exec_price
            cost = proceeds * commission
            net = proceeds - cost
            trade_ret = (net - entry_capital) / entry_capital
            trade_returns.append(trade_ret)
            capital += net
            position = 0
            shares = 0.0

        if position == 1:
            equity_curve[i] = capital + shares * closes[i]
        else:
            equity_curve[i] = capital

    returns = np.diff(equity_curve) / np.where(equity_curve[:-1] > 0, equity_curve[:-1], 1.0)
    return equity_curve, returns, trade_returns


def _stitch_equity_curves(windows: List[WindowResult]) -> List[float]:
    """Stitch test equity curves from all windows into one continuous curve.

    Each window starts at the ending value of the previous window.

    Args:
        windows: List of WindowResult objects.

    Returns:
        Combined equity curve as list of floats.
    """
    if not windows:
        return [100_000.0]

    combined = []
    scale = 1.0

    for w in windows:
        curve = w.test_equity_curve
        if not curve:
            continue

        if not combined:
            combined.extend(curve)
        else:
            # Scale this window to start where the previous ended
            prev_end = combined[-1]
            window_start = curve[0]
            if window_start > 0:
                scale = prev_end / window_start
            scaled = [v * scale for v in curve]
            combined.extend(scaled[1:])  # Skip first point (duplicate)

    return combined if combined else [100_000.0]


def _compute_max_drawdown(equity: np.ndarray) -> float:
    """Compute max drawdown from equity curve.

    Args:
        equity: Equity curve array.

    Returns:
        Max drawdown as positive fraction.
    """
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    drawdowns = np.where(peak > 0, (peak - equity) / peak, 0.0)
    return float(np.max(drawdowns))
