"""StrategyAgent — selects the best trading strategy based on market conditions.

This is the THIRD agent in the LangGraph workflow. It analyzes:
- Current market data (trending vs. oscillating)
- RAG context (fundamental analysis from SEC/news)
- Price history (volatility, trend strength)

Then selects one of four strategies:
- MomentumStrategy    (EMA crossover — smooth trending markets)
- MACDStrategy        (triple EMA — strong trending markets with momentum)
- RSIStrategy         (overbought/oversold — oscillating markets)
- MeanReversionStrategy (Z-score bands — high-volatility oscillating markets)

Selection logic (4-way decision tree):
    1. Compute trend strength: |price_change_pct| and 20-day SMA slope
    2. Compute volatility: rolling std of returns
    3. Count bullish/bearish signals from RAG context

    TRENDING (trend_strength > 5% OR sma_slope > 0.001):
        Strong trend + bullish fundamentals → MACD  (momentum confirmation)
        Moderate trend                      → Momentum (EMA crossover)

    OSCILLATING (no clear trend):
        High volatility (> 2.5%)            → MeanReversion (Z-score bands)
        Low/medium volatility               → RSI (overbought/oversold)

    ON RETRY: cycles through all 4 strategies in order

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
STRONG_TREND_THRESHOLD = 8.0     # % price change for MACD (stronger signal needed)
BULLISH_SIGNAL_THRESHOLD = 2     # Min bullish RAG signals to prefer MACD over Momentum

# Retry cycle: on each retry, rotate through all 4 strategies
_RETRY_CYCLE = ["momentum", "mean_reversion", "rsi", "macd"]


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

        # Read FinBERT sentiment score (from SentimentAgent, default neutral)
        sentiment_score = state.get("sentiment_score", 0.0)
        sentiment_label = state.get("sentiment_label", "NEUTRAL")

        logger.debug(
            "StrategyAgent | sentiment | score=%.3f | label=%s",
            sentiment_score, sentiment_label,
        )

        # On retry, rotate through all 4 strategies in a fixed cycle
        if retry_count > 0:
            cycle_idx = retry_count % len(_RETRY_CYCLE)
            next_strategy = _RETRY_CYCLE[cycle_idx]
            current = state.get("selected_strategy", "momentum")
            rationale = (
                f"Retry {retry_count}: Switching from {current} to {next_strategy} "
                f"after risk rejection. "
                f"Trend={trend_strength:.1f}% | Volatility={volatility:.3f}."
            )
            if next_strategy == "momentum":
                return _select_momentum(state, rationale)
            elif next_strategy == "mean_reversion":
                return _select_mean_reversion(state, rationale, volatility)
            elif next_strategy == "rsi":
                return _select_rsi(state, rationale)
            else:
                return _select_macd(state, rationale)

        # Primary selection logic (4-way decision tree + sentiment overlay)
        is_trending = (
            trend_strength > TREND_STRENGTH_THRESHOLD
            or abs(sma_slope) > SMA_SLOPE_THRESHOLD
        )
        is_strong_trend = trend_strength > STRONG_TREND_THRESHOLD
        is_high_volatility = volatility > VOLATILITY_THRESHOLD

        # Sentiment boosts: bullish sentiment lowers threshold for MACD,
        # bearish sentiment pushes toward MeanReversion even in trending markets
        is_sentiment_bullish = sentiment_score > 0.2
        is_sentiment_bearish = sentiment_score < -0.2

        if is_trending and not is_high_volatility:
            # Trending market — choose between MACD and Momentum
            if (is_strong_trend or is_sentiment_bullish) and bullish_signals >= BULLISH_SIGNAL_THRESHOLD:
                # Strong trend OR bullish sentiment + bullish fundamentals → MACD
                rationale = (
                    f"Strong trend detected: {trend_strength:.1f}% price change, "
                    f"SMA slope={sma_slope:.4f}. Volatility={volatility:.3f} (low). "
                    f"Sentiment: {sentiment_label} ({sentiment_score:+.2f}). "
                    f"MACD selected for momentum confirmation."
                )
                return _select_macd(state, rationale)
            elif is_sentiment_bearish and not is_strong_trend:
                # Mild trend but bearish sentiment → MeanReversion (wait for reversal)
                rationale = (
                    f"Mild trend ({trend_strength:.1f}%) but bearish sentiment "
                    f"({sentiment_label}: {sentiment_score:+.2f}). "
                    f"MeanReversion selected — sentiment suggests caution."
                )
                return _select_mean_reversion(state, rationale, volatility)
            else:
                # Moderate trend → Momentum (EMA crossover)
                rationale = (
                    f"Market is trending: {trend_strength:.1f}% price change, "
                    f"SMA slope={sma_slope:.4f}. Volatility={volatility:.3f} (low). "
                    f"Sentiment: {sentiment_label} ({sentiment_score:+.2f}). "
                    f"Momentum strategy selected."
                )
                return _select_momentum(state, rationale)

        else:
            # Oscillating market — choose between RSI and MeanReversion
            if is_high_volatility:
                rationale = (
                    f"Market is oscillating with HIGH volatility: "
                    f"trend_strength={trend_strength:.1f}%, volatility={volatility:.3f}. "
                    f"Sentiment: {sentiment_label} ({sentiment_score:+.2f}). "
                    f"MeanReversion (Z-score) selected for volatile oscillation."
                )
                return _select_mean_reversion(state, rationale, volatility)
            else:
                rationale = (
                    f"Market is oscillating with low volatility: "
                    f"trend_strength={trend_strength:.1f}%, volatility={volatility:.3f}. "
                    f"Sentiment: {sentiment_label} ({sentiment_score:+.2f}). "
                    f"RSI selected for overbought/oversold detection."
                )
                return _select_rsi(state, rationale)

    except Exception as e:
        logger.error("StrategyAgent | failed | ticker=%s | %s", ticker, e, exc_info=True)
        return {
            **state,
            "selected_strategy": "momentum",
            "strategy_params": {"short_window": 20, "long_window": 50},
            "strategy_rationale": f"Defaulted to momentum due to error: {e}",
            "error": f"StrategyAgent failed: {e}",
        }


def _select_rsi(state: TradingState, rationale: str) -> TradingState:
    """Return state with RSIStrategy selected.

    Args:
        state: Current state.
        rationale: Human-readable selection reason.

    Returns:
        Updated state with RSI strategy (period=14, oversold=30, overbought=70).
    """
    logger.info("StrategyAgent | selected=rsi | %s", rationale[:80])
    return {
        **state,
        "selected_strategy": "rsi",
        "strategy_params": {
            "period": 14,
            "oversold": 30,
            "overbought": 70,
        },
        "strategy_rationale": rationale,
    }


def _select_macd(state: TradingState, rationale: str) -> TradingState:
    """Return state with MACDStrategy selected.

    Args:
        state: Current state.
        rationale: Human-readable selection reason.

    Returns:
        Updated state with MACD strategy (12/26/9 — Appel's original values).
    """
    logger.info("StrategyAgent | selected=macd | %s", rationale[:80])
    return {
        **state,
        "selected_strategy": "macd",
        "strategy_params": {
            "fast": 12,
            "slow": 26,
            "signal_period": 9,
        },
        "strategy_rationale": rationale,
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
