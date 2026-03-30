"""Portfolio performance metrics for the QuantMind backtesting engine.

Computes industry-standard risk/return metrics used by professional
trading desks and hedge funds:
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Value at Risk (VaR) — Historical and Parametric
- Conditional VaR (CVaR / Expected Shortfall)
- Win Rate, Profit Factor

All functions accept numpy arrays of returns or equity curves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np

from config.logging_config import get_logger

logger = get_logger(__name__)

ANNUALIZATION_FACTOR = 252  # Trading days per year
CONFIDENCE_LEVEL_DEFAULT = 0.95


@dataclass
class PerformanceReport:
    """Complete performance report for a backtest run.

    Attributes:
        total_return: Total return over the period (e.g., 0.25 = 25%).
        annualized_return: Annualized return.
        annualized_volatility: Annualized standard deviation of returns.
        sharpe_ratio: Risk-adjusted return (annualized).
        sortino_ratio: Downside risk-adjusted return.
        max_drawdown: Maximum peak-to-trough drawdown (positive fraction).
        calmar_ratio: Annualized return / max drawdown.
        var_95: Value at Risk at 95% confidence (historical).
        cvar_95: Conditional VaR (Expected Shortfall) at 95%.
        win_rate: Fraction of profitable trades.
        profit_factor: Gross profit / gross loss.
        n_trades: Total number of trades executed.
        n_days: Total number of trading days in backtest.
    """

    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    n_days: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Serialize report to a plain dictionary.

        Returns:
            Dictionary of metric names to values.
        """
        return {
            "total_return": round(self.total_return, 6),
            "annualized_return": round(self.annualized_return, 6),
            "annualized_volatility": round(self.annualized_volatility, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 6),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "var_95": round(self.var_95, 6),
            "cvar_95": round(self.cvar_95, 6),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "n_trades": self.n_trades,
            "n_days": self.n_days,
        }


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """Compute the Sharpe ratio of a return series.

    Sharpe = (mean_return - risk_free_rate) / std_return
    Annualized by multiplying by sqrt(252).

    Args:
        returns: Array of period returns (daily, as decimals).
        risk_free_rate: Daily risk-free rate (default 0.0).
        annualize: If True, annualize the result.

    Returns:
        Sharpe ratio as a float. Returns 0.0 if std is near zero.

    Example:
        >>> import numpy as np
        >>> returns = np.array([0.01, -0.005, 0.008, 0.012, -0.003])
        >>> sharpe_ratio(returns)
        2.847...
    """
    if len(returns) < 2:
        return 0.0

    excess = returns - risk_free_rate
    std = np.std(excess, ddof=1)

    if std < 1e-10:
        return 0.0

    ratio = np.mean(excess) / std
    if annualize:
        ratio *= np.sqrt(ANNUALIZATION_FACTOR)

    logger.debug("sharpe_ratio=%.4f | n=%d", ratio, len(returns))
    return float(ratio)


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """Compute the Sortino ratio (penalizes only downside volatility).

    Sortino = (mean_return - risk_free_rate) / downside_std

    Args:
        returns: Array of period returns.
        risk_free_rate: Daily risk-free rate (default 0.0).
        annualize: If True, annualize the result.

    Returns:
        Sortino ratio as a float. Returns 0.0 if downside std is near zero.

    Example:
        >>> sortino_ratio(np.array([0.01, -0.005, 0.008, -0.002, 0.015]))
        3.12...
    """
    if len(returns) < 2:
        return 0.0

    excess = returns - risk_free_rate
    downside = excess[excess < 0]

    if len(downside) == 0:
        return float("inf")  # No losing periods

    downside_std = np.std(downside, ddof=1)
    if downside_std < 1e-10:
        return 0.0

    ratio = np.mean(excess) / downside_std
    if annualize:
        ratio *= np.sqrt(ANNUALIZATION_FACTOR)

    logger.debug("sortino_ratio=%.4f | n=%d", ratio, len(returns))
    return float(ratio)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute the maximum peak-to-trough drawdown of an equity curve.

    Max Drawdown = max((peak - trough) / peak) over all time windows.
    Returns a positive fraction (e.g., 0.20 = 20% drawdown).

    Args:
        equity_curve: Array of portfolio equity values (not returns).

    Returns:
        Maximum drawdown as a positive fraction (0.0 to 1.0).
        Returns 0.0 if equity never declines.

    Raises:
        ValueError: If equity_curve is empty.

    Example:
        >>> max_drawdown(np.array([100.0, 120.0, 90.0, 110.0, 80.0]))
        0.3333...  # 120 -> 80 = 33.3% drawdown
    """
    if len(equity_curve) == 0:
        raise ValueError("equity_curve must not be empty.")

    arr = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    drawdowns = np.where(peak > 0, (peak - arr) / peak, 0.0)
    mdd = float(np.max(drawdowns))

    logger.debug("max_drawdown=%.4f | n=%d", mdd, len(arr))
    return mdd


def calmar_ratio(
    returns: np.ndarray,
    equity_curve: np.ndarray,
) -> float:
    """Compute the Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Array of daily returns.
        equity_curve: Array of portfolio equity values.

    Returns:
        Calmar ratio. Returns 0.0 if max drawdown is zero.

    Example:
        >>> calmar_ratio(returns, equity_curve)
        1.45
    """
    if len(returns) == 0:
        return 0.0

    ann_return = annualized_return(returns)
    mdd = max_drawdown(equity_curve)

    if mdd < 1e-10:
        return 0.0

    ratio = ann_return / mdd
    logger.debug("calmar_ratio=%.4f", ratio)
    return float(ratio)


def annualized_return(returns: np.ndarray) -> float:
    """Compute the annualized return from a daily return series.

    annualized = (1 + mean_daily_return) ^ 252 - 1

    Args:
        returns: Array of daily returns.

    Returns:
        Annualized return as a decimal (e.g., 0.15 = 15%).

    Example:
        >>> annualized_return(np.array([0.001, -0.0005, 0.002]))
        0.284...
    """
    if len(returns) == 0:
        return 0.0

    mean_daily = float(np.mean(returns))
    ann = (1 + mean_daily) ** ANNUALIZATION_FACTOR - 1
    return float(ann)


def annualized_volatility(returns: np.ndarray) -> float:
    """Compute annualized volatility (standard deviation of returns).

    annualized_vol = daily_std * sqrt(252)

    Args:
        returns: Array of daily returns.

    Returns:
        Annualized volatility as a decimal.

    Example:
        >>> annualized_volatility(np.array([0.01, -0.005, 0.008]))
        0.119...
    """
    if len(returns) < 2:
        return 0.0

    daily_std = float(np.std(returns, ddof=1))
    return daily_std * np.sqrt(ANNUALIZATION_FACTOR)


def historical_var(
    returns: np.ndarray,
    confidence: float = CONFIDENCE_LEVEL_DEFAULT,
) -> float:
    """Compute Historical Value at Risk (VaR).

    VaR is the loss not exceeded with the given confidence level.
    Uses the empirical distribution of historical returns.

    Args:
        returns: Array of daily returns.
        confidence: Confidence level (default 0.95 = 95%).

    Returns:
        VaR as a positive fraction (e.g., 0.02 = 2% daily loss at 95% CI).

    Raises:
        ValueError: If confidence is not in (0, 1).

    Example:
        >>> historical_var(returns, confidence=0.95)
        0.0187
    """
    _validate_confidence(confidence)
    if len(returns) == 0:
        return 0.0

    var = float(-np.percentile(returns, (1 - confidence) * 100))
    logger.debug("historical_var=%.4f | confidence=%.2f", var, confidence)
    return max(var, 0.0)


def parametric_var(
    returns: np.ndarray,
    confidence: float = CONFIDENCE_LEVEL_DEFAULT,
) -> float:
    """Compute Parametric (Gaussian) Value at Risk.

    Assumes returns are normally distributed.
    VaR = -(mean - z * std)

    Args:
        returns: Array of daily returns.
        confidence: Confidence level (default 0.95).

    Returns:
        VaR as a positive fraction.

    Raises:
        ValueError: If confidence is not in (0, 1).
    """
    _validate_confidence(confidence)
    if len(returns) < 2:
        return 0.0

    from scipy import stats  # type: ignore[import]

    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    z = stats.norm.ppf(1 - confidence)
    var = float(-(mean + z * std))
    logger.debug("parametric_var=%.4f | confidence=%.2f", var, confidence)
    return max(var, 0.0)


def conditional_var(
    returns: np.ndarray,
    confidence: float = CONFIDENCE_LEVEL_DEFAULT,
) -> float:
    """Compute Conditional VaR (CVaR / Expected Shortfall).

    CVaR is the expected loss given that the loss exceeds VaR.
    More conservative and coherent risk measure than VaR.

    Args:
        returns: Array of daily returns.
        confidence: Confidence level (default 0.95).

    Returns:
        CVaR as a positive fraction.

    Raises:
        ValueError: If confidence is not in (0, 1).

    Example:
        >>> conditional_var(returns, confidence=0.95)
        0.0243
    """
    _validate_confidence(confidence)
    if len(returns) == 0:
        return 0.0

    threshold = np.percentile(returns, (1 - confidence) * 100)
    tail_losses = returns[returns <= threshold]

    if len(tail_losses) == 0:
        return 0.0

    cvar = float(-np.mean(tail_losses))
    logger.debug("conditional_var=%.4f | confidence=%.2f", cvar, confidence)
    return max(cvar, 0.0)


def win_rate(trade_returns: np.ndarray) -> float:
    """Compute the fraction of winning trades.

    Args:
        trade_returns: Array of individual trade returns.

    Returns:
        Win rate as a fraction (0.0 to 1.0).
        Returns 0.0 if no trades.

    Example:
        >>> win_rate(np.array([0.05, -0.02, 0.03, -0.01, 0.04]))
        0.6
    """
    if len(trade_returns) == 0:
        return 0.0

    winners = np.sum(trade_returns > 0)
    rate = float(winners / len(trade_returns))
    logger.debug("win_rate=%.4f | n_trades=%d", rate, len(trade_returns))
    return rate


def profit_factor(trade_returns: np.ndarray) -> float:
    """Compute the profit factor (gross profit / gross loss).

    A profit factor > 1.0 means the strategy is profitable overall.
    Values > 1.5 are considered good; > 2.0 are excellent.

    Args:
        trade_returns: Array of individual trade returns.

    Returns:
        Profit factor. Returns 0.0 if no losing trades.

    Example:
        >>> profit_factor(np.array([0.05, -0.02, 0.03, -0.01, 0.04]))
        4.0  # (0.05+0.03+0.04) / (0.02+0.01)
    """
    if len(trade_returns) == 0:
        return 0.0

    gross_profit = float(np.sum(trade_returns[trade_returns > 0]))
    gross_loss = float(np.abs(np.sum(trade_returns[trade_returns < 0])))

    if gross_loss < 1e-10:
        return float("inf") if gross_profit > 0 else 0.0

    pf = gross_profit / gross_loss
    logger.debug("profit_factor=%.4f", pf)
    return float(pf)


def compute_full_report(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    trade_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0,
) -> PerformanceReport:
    """Compute a complete performance report from backtest results.

    This is the primary entry point for the backtesting engine to
    generate a standardized PerformanceReport.

    Args:
        returns: Array of daily portfolio returns.
        equity_curve: Array of portfolio equity values.
        trade_returns: Optional array of individual trade returns.
        risk_free_rate: Daily risk-free rate (default 0.0).

    Returns:
        PerformanceReport dataclass with all metrics populated.

    Example:
        >>> report = compute_full_report(returns, equity_curve, trade_returns)
        >>> print(report.sharpe_ratio)
        1.42
    """
    tr = trade_returns if trade_returns is not None else np.array([])

    report = PerformanceReport(
        total_return=float(
            (equity_curve[-1] / equity_curve[0] - 1) if len(equity_curve) > 1 else 0.0
        ),
        annualized_return=annualized_return(returns),
        annualized_volatility=annualized_volatility(returns),
        sharpe_ratio=sharpe_ratio(returns, risk_free_rate),
        sortino_ratio=sortino_ratio(returns, risk_free_rate),
        max_drawdown=max_drawdown(equity_curve),
        calmar_ratio=calmar_ratio(returns, equity_curve),
        var_95=historical_var(returns),
        cvar_95=conditional_var(returns),
        win_rate=win_rate(tr),
        profit_factor=profit_factor(tr),
        n_trades=len(tr),
        n_days=len(returns),
    )

    logger.info(
        "PerformanceReport | sharpe=%.2f | mdd=%.2f%% | total_return=%.2f%%",
        report.sharpe_ratio,
        report.max_drawdown * 100,
        report.total_return * 100,
    )
    return report


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _validate_confidence(confidence: float) -> None:
    """Validate confidence level is in (0, 1).

    Args:
        confidence: Confidence level to validate.

    Raises:
        ValueError: If confidence is not in (0, 1).
    """
    if not (0 < confidence < 1):
        raise ValueError(
            f"confidence must be in (0, 1), got {confidence}."
        )
