"""Sliding window utilities for O(n) rolling financial metrics.

Used in the backtesting engine to compute:
- Rolling mean (Simple Moving Average)
- Rolling standard deviation (volatility)
- Rolling Sharpe ratio
- Exponential Moving Average (EMA)
- Rolling max drawdown

All functions operate on plain Python lists or numpy arrays and return
numpy arrays for downstream use in strategies and metrics.

Time Complexity: O(n) for all rolling computations.
Space Complexity: O(n) output arrays.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from config.logging_config import get_logger

logger = get_logger(__name__)

# Type alias for numeric sequences
NumericSequence = Union[List[float], np.ndarray]

ANNUALIZATION_FACTOR = 252  # Trading days per year


def rolling_mean(
    values: NumericSequence,
    window: int,
) -> np.ndarray:
    """Compute Simple Moving Average (SMA) using a sliding window.

    Positions with fewer than `window` data points are filled with NaN.

    Args:
        values: Sequence of numeric values (e.g., closing prices).
        window: Look-back window size in periods.

    Returns:
        NumPy array of rolling means, same length as input.

    Raises:
        ValueError: If window < 1 or values is empty.

    Example:
        >>> rolling_mean([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
        array([nan, nan, 2.0, 3.0, 4.0])
    """
    _validate_window(values, window)
    arr = np.asarray(values, dtype=float)
    result = np.full(len(arr), np.nan)

    cumsum = np.cumsum(arr)
    # From index window-1 onward we have a full window
    result[window - 1 :] = (
        cumsum[window - 1 :] - np.concatenate([[0.0], cumsum[: len(arr) - window]])
    ) / window

    logger.debug("rolling_mean | n=%d | window=%d", len(arr), window)
    return result


def rolling_std(
    values: NumericSequence,
    window: int,
    ddof: int = 1,
) -> np.ndarray:
    """Compute rolling standard deviation (volatility) using a sliding window.

    Args:
        values: Sequence of numeric values.
        window: Look-back window size in periods.
        ddof: Delta degrees of freedom (1 = sample std, 0 = population std).

    Returns:
        NumPy array of rolling standard deviations, NaN for incomplete windows.

    Raises:
        ValueError: If window < 2 (std requires at least 2 points).

    Example:
        >>> rolling_std([1.0, 2.0, 3.0, 4.0, 5.0], window=3)
        array([nan, nan, 1.0, 1.0, 1.0])
    """
    if window < 2:
        raise ValueError(f"window must be >= 2 for std, got {window}.")
    _validate_window(values, window)

    arr = np.asarray(values, dtype=float)
    n = len(arr)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_slice = arr[i - window + 1 : i + 1]
        result[i] = np.std(window_slice, ddof=ddof)

    logger.debug("rolling_std | n=%d | window=%d | ddof=%d", n, window, ddof)
    return result


def rolling_sharpe(
    returns: NumericSequence,
    window: int,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> np.ndarray:
    """Compute rolling Sharpe ratio over a sliding window.

    Sharpe = (mean_return - risk_free_rate) / std_return
    Annualized by multiplying by sqrt(252) when annualize=True.

    Args:
        returns: Sequence of period returns (e.g., daily % returns as decimals).
        window: Look-back window size in periods.
        risk_free_rate: Daily risk-free rate (default 0.0).
        annualize: If True, annualize the Sharpe ratio by sqrt(252).

    Returns:
        NumPy array of rolling Sharpe ratios, NaN for incomplete windows
        or zero-volatility windows.

    Example:
        >>> import numpy as np
        >>> returns = np.random.normal(0.001, 0.01, 300)
        >>> sharpe = rolling_sharpe(returns, window=60)
    """
    _validate_window(returns, window)
    arr = np.asarray(returns, dtype=float)
    n = len(arr)
    result = np.full(n, np.nan)
    scale = np.sqrt(ANNUALIZATION_FACTOR) if annualize else 1.0

    for i in range(window - 1, n):
        window_slice = arr[i - window + 1 : i + 1]
        mean_r = np.mean(window_slice) - risk_free_rate
        std_r = np.std(window_slice, ddof=1)
        if std_r > 1e-10:  # Avoid division by zero
            result[i] = (mean_r / std_r) * scale

    logger.debug(
        "rolling_sharpe | n=%d | window=%d | annualize=%s", n, window, annualize
    )
    return result


def exponential_moving_average(
    values: NumericSequence,
    span: int,
) -> np.ndarray:
    """Compute Exponential Moving Average (EMA) using smoothing factor.

    alpha = 2 / (span + 1)
    EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}

    Args:
        values: Sequence of numeric values (e.g., closing prices).
        span: EMA span (e.g., 12 for 12-day EMA).

    Returns:
        NumPy array of EMA values, same length as input.

    Raises:
        ValueError: If span < 1 or values is empty.

    Example:
        >>> ema = exponential_moving_average([10.0, 11.0, 12.0, 11.5, 13.0], span=3)
    """
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span}.")
    if len(values) == 0:
        raise ValueError("values must not be empty.")

    arr = np.asarray(values, dtype=float)
    alpha = 2.0 / (span + 1)
    result = np.empty(len(arr))
    result[0] = arr[0]

    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]

    logger.debug("EMA | n=%d | span=%d | alpha=%.4f", len(arr), span, alpha)
    return result


def rolling_max_drawdown(
    equity_curve: NumericSequence,
    window: int,
) -> np.ndarray:
    """Compute rolling maximum drawdown over a sliding window.

    Max Drawdown = (peak - trough) / peak within the window.
    Returns values as positive fractions (e.g., 0.15 = 15% drawdown).

    Args:
        equity_curve: Sequence of portfolio equity values (not returns).
        window: Look-back window size in periods.

    Returns:
        NumPy array of rolling max drawdown values (0.0 to 1.0),
        NaN for incomplete windows.

    Example:
        >>> equity = [100.0, 110.0, 105.0, 95.0, 100.0]
        >>> rolling_max_drawdown(equity, window=3)
        array([nan, nan, 0.0454..., 0.1363..., 0.0952...])
    """
    _validate_window(equity_curve, window)
    arr = np.asarray(equity_curve, dtype=float)
    n = len(arr)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_slice = arr[i - window + 1 : i + 1]
        peak = np.maximum.accumulate(window_slice)
        drawdowns = (peak - window_slice) / np.where(peak > 0, peak, 1.0)
        result[i] = np.max(drawdowns)

    logger.debug("rolling_max_drawdown | n=%d | window=%d", n, window)
    return result


def compute_returns(prices: NumericSequence) -> np.ndarray:
    """Compute simple period-over-period returns from a price series.

    return_t = (price_t - price_{t-1}) / price_{t-1}

    Args:
        prices: Sequence of asset prices (must have at least 2 elements).

    Returns:
        NumPy array of returns, length = len(prices) - 1.
        First element corresponds to the return from prices[0] to prices[1].

    Raises:
        ValueError: If prices has fewer than 2 elements.

    Example:
        >>> compute_returns([100.0, 105.0, 102.0, 108.0])
        array([ 0.05  , -0.02857,  0.05882])
    """
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 2:
        raise ValueError(
            f"prices must have at least 2 elements, got {len(arr)}."
        )
    returns = np.diff(arr) / arr[:-1]
    logger.debug("compute_returns | n_prices=%d | n_returns=%d", len(arr), len(returns))
    return returns


def compute_log_returns(prices: NumericSequence) -> np.ndarray:
    """Compute log returns from a price series.

    log_return_t = ln(price_t / price_{t-1})

    Args:
        prices: Sequence of asset prices (must have at least 2 elements).

    Returns:
        NumPy array of log returns, length = len(prices) - 1.

    Raises:
        ValueError: If prices has fewer than 2 elements or contains non-positive values.

    Example:
        >>> compute_log_returns([100.0, 105.0, 102.0])
        array([ 0.04879,  -0.02899])
    """
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 2:
        raise ValueError(
            f"prices must have at least 2 elements, got {len(arr)}."
        )
    if np.any(arr <= 0):
        raise ValueError("All prices must be positive for log returns.")

    log_returns = np.diff(np.log(arr))
    logger.debug(
        "compute_log_returns | n_prices=%d | n_returns=%d", len(arr), len(log_returns)
    )
    return log_returns


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _validate_window(values: NumericSequence, window: int) -> None:
    """Validate that window size is valid for the given data.

    Args:
        values: Input data sequence.
        window: Window size to validate.

    Raises:
        ValueError: If window < 1 or window > len(values).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    if len(values) == 0:
        raise ValueError("values must not be empty.")
    if window > len(values):
        raise ValueError(
            f"window ({window}) cannot exceed data length ({len(values)})."
        )
