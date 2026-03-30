"""RiskAgent — evaluates strategy risk and approves or rejects.

This is the FIFTH agent in the LangGraph workflow. It:
1. Evaluates backtest results against risk limits
2. Computes a composite risk score (0-10)
3. Approves if all limits are satisfied
4. Rejects and triggers retry (back to StrategyAgent) if limits breached

Risk limits (configurable):
    - Sharpe ratio >= 0.5 (minimum acceptable risk-adjusted return)
    - Max drawdown <= 25% (maximum acceptable loss)
    - VaR 95% <= 5% (maximum daily loss at 95% confidence)
    - Win rate >= 30% (minimum trade success rate)

Retry logic:
    If risk is rejected AND retry_count < MAX_RETRIES:
        → Return to StrategyAgent with incremented retry_count
        → StrategyAgent will switch to the other strategy
    If retry_count >= MAX_RETRIES:
        → Approve with warning (best available strategy)

LangGraph node contract:
    Input:  TradingState with backtest_results
    Output: TradingState with risk_metrics, risk_approved
            Conditional edge: approved → ExplainerAgent
                              rejected → StrategyAgent (retry)
"""

from __future__ import annotations

from config.logging_config import get_logger
from graph.state import RiskMetrics, TradingState

logger = get_logger(__name__)

# Risk limits
MIN_SHARPE = 0.5
MAX_DRAWDOWN = 0.25
MAX_VAR_95 = 0.05
MIN_WIN_RATE = 0.30
MAX_RETRIES = 3


async def risk_agent(state: TradingState) -> TradingState:
    """Evaluate strategy risk and approve or reject. LangGraph node function.

    Computes a composite risk score and checks against limits.
    If rejected, increments retry_count for StrategyAgent to switch strategy.

    Args:
        state: TradingState with backtest_results populated.

    Returns:
        Updated TradingState with risk_metrics and risk_approved.
        risk_approved=True → workflow continues to ExplainerAgent.
        risk_approved=False → workflow retries StrategyAgent.

    Example:
        >>> state = await risk_agent(state)
        >>> state["risk_approved"]
        True
        >>> state["risk_metrics"]["risk_level"]
        'MEDIUM'
    """
    ticker = state.get("ticker", "")
    backtest = state.get("backtest_results", {})
    retry_count = state.get("retry_count", 0)

    sharpe = backtest.get("sharpe_ratio", 0.0)
    max_dd = backtest.get("max_drawdown", 1.0)
    var_95 = backtest.get("var_95", 1.0)
    win_rate = backtest.get("win_rate", 0.0)
    total_return = backtest.get("total_return", 0.0)

    logger.info(
        "RiskAgent | evaluating | ticker=%s | sharpe=%.2f | mdd=%.1f%% | "
        "var=%.2f%% | win_rate=%.1f%% | retry=%d",
        ticker, sharpe, max_dd * 100, var_95 * 100, win_rate * 100, retry_count,
    )

    # Compute composite risk score (0-10, lower = safer)
    risk_score = _compute_risk_score(sharpe, max_dd, var_95, win_rate)
    risk_level = _get_risk_level(risk_score)

    # Check individual risk limits
    violations: list[str] = []

    if sharpe < MIN_SHARPE:
        violations.append(f"Sharpe {sharpe:.2f} < {MIN_SHARPE} minimum")

    if max_dd > MAX_DRAWDOWN:
        violations.append(f"Max drawdown {max_dd:.1%} > {MAX_DRAWDOWN:.0%} limit")

    if var_95 > MAX_VAR_95:
        violations.append(f"VaR 95% {var_95:.2%} > {MAX_VAR_95:.0%} limit")

    if win_rate < MIN_WIN_RATE and backtest.get("n_trades", 0) > 5:
        violations.append(f"Win rate {win_rate:.1%} < {MIN_WIN_RATE:.0%} minimum")

    risk_metrics = RiskMetrics(
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        var_95=var_95,
        risk_score=round(risk_score, 2),
        risk_level=risk_level,
        risk_approved=len(violations) == 0,
        rejection_reason="; ".join(violations) if violations else "",
    )

    if violations:
        if retry_count >= MAX_RETRIES:
            # Max retries reached — approve with warning
            logger.warning(
                "RiskAgent | max retries reached | approving with warning | "
                "ticker=%s | violations=%s",
                ticker, violations,
            )
            risk_metrics["risk_approved"] = True
            risk_metrics["rejection_reason"] = (
                f"Approved after {MAX_RETRIES} retries (best available). "
                f"Violations: {'; '.join(violations)}"
            )
            return {
                **state,
                "risk_metrics": risk_metrics,
                "risk_approved": True,
                "retry_count": retry_count,
            }
        else:
            # Reject and trigger retry
            logger.warning(
                "RiskAgent | rejected | ticker=%s | retry=%d | violations=%s",
                ticker, retry_count + 1, violations,
            )
            return {
                **state,
                "risk_metrics": risk_metrics,
                "risk_approved": False,
                "retry_count": retry_count + 1,
            }
    else:
        logger.info(
            "RiskAgent | approved | ticker=%s | score=%.1f | level=%s | "
            "sharpe=%.2f | mdd=%.1f%%",
            ticker, risk_score, risk_level, sharpe, max_dd * 100,
        )
        return {
            **state,
            "risk_metrics": risk_metrics,
            "risk_approved": True,
        }


def _compute_risk_score(
    sharpe: float,
    max_dd: float,
    var_95: float,
    win_rate: float,
) -> float:
    """Compute composite risk score (0-10, lower = safer).

    Weights:
        - Sharpe ratio:   30% (higher = lower risk)
        - Max drawdown:   40% (lower = lower risk)
        - VaR 95%:        20% (lower = lower risk)
        - Win rate:       10% (higher = lower risk)

    Args:
        sharpe: Sharpe ratio.
        max_dd: Maximum drawdown (0-1).
        var_95: Value at Risk at 95% (0-1).
        win_rate: Win rate (0-1).

    Returns:
        Risk score from 0 (safest) to 10 (riskiest).
    """
    # Normalize each metric to 0-10 scale
    sharpe_score = max(0.0, min(10.0, (2.0 - sharpe) * 5))  # Sharpe 2+ = 0 risk
    dd_score = min(10.0, max_dd * 40)                         # 25% dd = 10 risk
    var_score = min(10.0, var_95 * 200)                       # 5% VaR = 10 risk
    win_score = max(0.0, min(10.0, (0.5 - win_rate) * 20))   # 50%+ win = 0 risk

    # Weighted composite
    return (
        0.30 * sharpe_score
        + 0.40 * dd_score
        + 0.20 * var_score
        + 0.10 * win_score
    )


def _get_risk_level(risk_score: float) -> str:
    """Convert numeric risk score to human-readable level.

    Args:
        risk_score: Composite risk score (0-10).

    Returns:
        Risk level string: 'LOW', 'MEDIUM', or 'HIGH'.
    """
    if risk_score < 3.0:
        return "LOW"
    elif risk_score < 6.0:
        return "MEDIUM"
    else:
        return "HIGH"


def should_retry(state: TradingState) -> str:
    """LangGraph conditional edge function.

    Determines whether to retry StrategyAgent or proceed to ExplainerAgent.

    Args:
        state: Current TradingState after RiskAgent.

    Returns:
        'retry' → route back to StrategyAgent
        'approved' → route forward to ExplainerAgent

    Example:
        >>> # Used in workflow.py as conditional edge
        >>> workflow.add_conditional_edges("risk_agent", should_retry, {
        ...     "retry": "strategy_agent",
        ...     "approved": "explainer_agent",
        ... })
    """
    if state.get("risk_approved", False):
        return "approved"
    return "retry"
