"""StrategyAgent — selects the best trading strategy based on market conditions.

This is the THIRD agent in the LangGraph workflow. It analyzes:
- Current market data (trending vs. oscillating)
- RAG context (fundamental analysis from SEC/news)
- Price history (volatility, trend strength)

Then selects either:
- MomentumStrategy (for trending markets)
- MeanReversionStrategy (for oscillating/sideways markets)

And determines optimal hyperparameters for the selected strategy.

Selection logic:
    1. Compute trend strength: |price_change_pct| and 20-day SMA slope
    2. Compute volatility: rolling std of returns
    3. If trend_strength > threshold AND volatility < threshold → Momentum
    4. Else → MeanReversion
    5. Adjust hyperparameters based on volatility regime

LangGraph node contract:
    Input:  TradingState with market_data, price_history, rag_context
    Output: TradingState with selected_strategy, strategy_params, strategy_rationale
"""

from __future__ import annotations

import numpy as np

from config.logging_config import get_logger
from graph.state import TradingState

logger = get_logger(__name__)

# Strategy selection thresholds
TREND_STRENGTH_THRESHOLD = 5.0   # % price change to consider trending
VOLATILITY_THRESHOLD = 0.025     # Daily std > 2.5% = high volatility
SMA_SLOPE_THRESHOLD = 0.001      # Normalized SMA slope for trend detection


async def strategy_agent(state: TradingState) -> TradingState:
    """Select the best trading strategy. LangGraph node function.

    Analyzes market conditions and selects Momentum or MeanReversion
    strategy with optimized hyperparameters.

    Args:
        state: TradingState with market_data and price_history.

    Returns:
        Updated TradingState with selected_strategy, strategy_params,
        strategy_rationale populated.

    Example:
        >>> state = await strategy_agent(state)
        >>> state["selected_strategy"]
        'momentum'
        >>> state["strategy_params"]
        {"short_window": 20, "long_window": 50}
    """
    ticker = state.get("ticker", "")
    market_data = state.get("market_data", {})
    price_history = state.get("price_history", [])
    rag_context = state.get("rag_context", "")
    retry_count = state.get("retry_count", 0)

    logger.info(
        "StrategyAgent | analyzing | ticker=%s | retry=%d", ticker, retry_count
    )

    try:
        # Extract price series from history
        closes = [bar["close"] for bar in price_history if "close" in bar]

        if len(closes) < 50:
            # Not enough data — default to momentum
            logger.warning(
                "StrategyAgent | insufficient price history | n=%d | defaulting to momentum",
                len(closes),
            )
            return _select_momentum(state, "Insufficient price history — defaulting to momentum.")

        closes_arr = np.array(closes, dtype=float)

        # Compute market regime indicators
        trend_strength = abs(market_data.get("price_change_pct", 0.0))
        returns = np.diff(closes_arr) / closes_arr[:-1]
        volatility = float(np.std(returns[-20:], ddof=1)) if len(returns) >= 20 else 0.02

        # SMA slope: normalized slope of 20-day SMA
        sma_20 = np.convolve(closes_arr, np.ones(20) / 20, mode="valid")
        if len(sma_20) >= 2:
            sma_slope = (sma_20[-1] - sma_20[-5]) / (sma_20[-5] * 5) if sma_20[-5] > 0 else 0.0
        else:
            sma_slope = 0.0

        # Check if RAG context mentions bearish/bullish signals
        rag_lower = rag_context.lower()
        bullish_signals = sum(1 for w in ["growth", "record", "beat", "strong", "increase", "rose"] if w in rag_lower)
        bearish_signals = sum(1 for w in ["decline", "fell", "miss", "weak", "decrease", "dropped"] if w in rag_lower)

        logger.debug(
            "StrategyAgent | regime | trend=%.2f%% | vol=%.4f | sma_slope=%.6f | "
            "bullish=%d | bearish=%d",
            trend_strength, volatility, sma_slope, bullish_signals, bearish_signals,
        )

        # On retry, switch strategy
        if retry_count > 0:
            current = state.get("selected_strategy", "momentum")
            if current == "momentum":
                rationale = (
                    f"Retry {retry_count}: Switching from Momentum to MeanReversion "
                    f"after risk rejection. Volatility={volatility:.3f}."
                )
                return _select_mean_reversion(state, rationale, volatility)
            else:
                rationale = (
                    f"Retry {retry_count}: Switching from MeanReversion to Momentum "
                    f"after risk rejection. Trend strength={trend_strength:.1f}%."
                )
                return _select_momentum(state, rationale)

        # Primary selection logic
        is_trending = (
            trend_strength > TREND_STRENGTH_THRESHOLD
            or abs(sma_slope) > SMA_SLOPE_THRESHOLD
        )
        is_high_volatility = volatility > VOLATILITY_THRESHOLD

        if is_trending and not is_high_volatility:
            rationale = (
                f"Market is trending: {trend_strength:.1f}% price change, "
                f"SMA slope={sma_slope:.4f}. "
                f"Volatility={volatility:.3f} (low). "
                f"Momentum strategy selected."
            )
            if bullish_signals > bearish_signals:
                rationale += f" Fundamentals bullish ({bullish_signals} positive signals)."
            return _select_momentum(state, rationale)

        else:
            rationale = (
                f"Market is oscillating: trend_strength={trend_strength:.1f}%, "
                f"volatility={volatility:.3f}. "
                f"MeanReversion strategy selected."
            )
            if bearish_signals > bullish_signals:
                rationale += f" Fundamentals bearish ({bearish_signals} negative signals)."
            return _select_mean_reversion(state, rationale, volatility)

    except Exception as e:
        logger.error("StrategyAgent | failed | ticker=%s | %s", ticker, e, exc_info=True)
        return {
            **state,
            "selected_strategy": "momentum",
            "strategy_params": {"short_window": 20, "long_window": 50},
            "strategy_rationale": f"Defaulted to momentum due to error: {e}",
            "error": f"StrategyAgent failed: {e}",
        }


def _select_momentum(state: TradingState, rationale: str) -> TradingState:
    """Return state with MomentumStrategy selected.

    Args:
        state: Current state.
        rationale: Human-readable selection reason.

    Returns:
        Updated state with momentum strategy.
    """
    logger.info("StrategyAgent | selected=momentum | %s", rationale[:80])
    return {
        **state,
        "selected_strategy": "momentum",
        "strategy_params": {
            "short_window": 20,
            "long_window": 50,
            "use_ema": 0,
        },
        "strategy_rationale": rationale,
    }


def _select_mean_reversion(
    state: TradingState, rationale: str, volatility: float = 0.02
) -> TradingState:
    """Return state with MeanReversionStrategy selected.

    Adjusts z_threshold based on volatility:
    - High volatility → wider threshold (2.5) to avoid false signals
    - Low volatility → tighter threshold (1.5) for more signals

    Args:
        state: Current state.
        rationale: Human-readable selection reason.
        volatility: Daily return std for threshold adjustment.

    Returns:
        Updated state with mean reversion strategy.
    """
    z_threshold = 2.5 if volatility > VOLATILITY_THRESHOLD else 1.5
    logger.info("StrategyAgent | selected=mean_reversion | z=%.1f | %s", z_threshold, rationale[:80])
    return {
        **state,
        "selected_strategy": "mean_reversion",
        "strategy_params": {
            "window": 20,
            "z_threshold": z_threshold,
            "exit_threshold": 0.0,
        },
        "strategy_rationale": rationale,
    }
