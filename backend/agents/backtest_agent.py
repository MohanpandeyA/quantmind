"""BacktestAgent — runs the selected strategy on historical price data.

This is the FOURTH agent in the LangGraph workflow. It:
1. Instantiates the strategy selected by StrategyAgent
2. Runs the Backtester on real historical data (yfinance)
3. Computes full performance metrics (Sharpe, VaR, Drawdown, etc.)
4. Stores the equity curve for charting

Connects Phase 1 (Backtesting Engine) to Phase 3 (LangGraph Agents).

LangGraph node contract:
    Input:  TradingState with selected_strategy, strategy_params, ticker, dates
    Output: TradingState with backtest_results, equity_curve populated
"""

from __future__ import annotations

import numpy as np

from config.logging_config import get_logger
from engine.backtester import Backtester, BacktestConfig
from engine.strategies.base_strategy import StrategyConfig
from engine.strategies.mean_reversion import MeanReversionStrategy
from engine.strategies.momentum import MomentumStrategy
from graph.state import BacktestResults, TradingState

logger = get_logger(__name__)


async def backtest_agent(state: TradingState) -> TradingState:
    """Run backtest for the selected strategy. LangGraph node function.

    Instantiates the strategy from StrategyAgent's selection and runs
    the full backtesting pipeline using Phase 1's Backtester.

    Args:
        state: TradingState with selected_strategy, strategy_params,
               ticker, start_date, end_date.

    Returns:
        Updated TradingState with backtest_results and equity_curve.

    Example:
        >>> state = await backtest_agent(state)
        >>> state["backtest_results"]["sharpe_ratio"]
        1.42
        >>> state["backtest_results"]["max_drawdown"]
        0.12
    """
    ticker = state.get("ticker", "")
    selected_strategy = state.get("selected_strategy", "momentum")
    strategy_params = state.get("strategy_params", {})
    start_date = state.get("start_date", "2022-01-01")
    end_date = state.get("end_date", "2024-12-31")

    logger.info(
        "BacktestAgent | running | ticker=%s | strategy=%s | %s → %s",
        ticker, selected_strategy, start_date, end_date,
    )

    try:
        # Build strategy config from StrategyAgent's params
        strategy_config = StrategyConfig(
            initial_capital=100_000.0,
            position_size=1.0,
            commission=0.001,
            params=strategy_params,
        )

        # Instantiate the selected strategy
        if selected_strategy == "momentum":
            strategy = MomentumStrategy(strategy_config)
        elif selected_strategy == "mean_reversion":
            strategy = MeanReversionStrategy(strategy_config)
        else:
            logger.warning(
                "BacktestAgent | unknown strategy '%s' — defaulting to momentum",
                selected_strategy,
            )
            strategy = MomentumStrategy(strategy_config)

        # Run backtest using Phase 1's Backtester
        bt_config = BacktestConfig(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            long_only=True,
            use_segment_tree=True,
        )

        backtester = Backtester(bt_config, strategy)
        result, report = backtester.run()

        # Build BacktestResults from PerformanceReport
        backtest_results = BacktestResults(
            strategy_name=result.strategy_name,
            total_return=round(report.total_return, 4),
            annualized_return=round(report.annualized_return, 4),
            sharpe_ratio=round(report.sharpe_ratio, 3),
            sortino_ratio=round(report.sortino_ratio, 3),
            max_drawdown=round(report.max_drawdown, 4),
            calmar_ratio=round(report.calmar_ratio, 3),
            var_95=round(report.var_95, 4),
            cvar_95=round(report.cvar_95, 4),
            win_rate=round(report.win_rate, 3),
            profit_factor=round(report.profit_factor, 3),
            n_trades=report.n_trades,
            n_days=report.n_days,
            start_date=start_date,
            end_date=end_date,
        )

        # Equity curve as list of floats for charting
        equity_curve = result.equity_curve.tolist()

        logger.info(
            "BacktestAgent | complete | ticker=%s | strategy=%s | "
            "sharpe=%.2f | return=%.1f%% | mdd=%.1f%% | trades=%d",
            ticker, selected_strategy,
            report.sharpe_ratio,
            report.total_return * 100,
            report.max_drawdown * 100,
            report.n_trades,
        )

        return {
            **state,
            "backtest_results": backtest_results,
            "equity_curve": equity_curve,
        }

    except Exception as e:
        logger.error(
            "BacktestAgent | failed | ticker=%s | strategy=%s | %s",
            ticker, selected_strategy, e, exc_info=True,
        )
        # Return neutral results on failure
        return {
            **state,
            "backtest_results": BacktestResults(
                strategy_name=selected_strategy,
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                var_95=0.0,
                cvar_95=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                n_trades=0,
                n_days=0,
                start_date=start_date,
                end_date=end_date,
            ),
            "equity_curve": [],
            "error": f"BacktestAgent failed: {e}",
        }
