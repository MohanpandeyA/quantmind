"""Unit tests for Phase 3 LangGraph agents.

Tests cover:
- TradingState: creation, field defaults, create_initial_state()
- StrategyAgent: momentum vs mean_reversion selection, retry logic
- RiskAgent: approval, rejection, retry, max retries, risk scoring
- ExplainerAgent: signal parsing, fallback explanation, signal_from_backtest
- API Schemas: AnalysisRequest validation, AnalysisResponse.from_trading_state()

All tests use mocks — no real API calls, no yfinance, no Groq.
"""

from __future__ import annotations

import asyncio
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from graph.state import (
    TradingState,
    BacktestResults,
    MarketData,
    RiskMetrics,
    create_initial_state,
)
from agents.strategy_agent import (
    strategy_agent,
    _select_momentum,
    _select_mean_reversion,
)
from agents.risk_agent import (
    risk_agent,
    should_retry,
    _compute_risk_score,
    _get_risk_level,
)
from agents.explainer_agent import (
    _parse_signal,
    _signal_from_backtest,
    _generate_fallback_explanation,
)
from api.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    BacktestResultsResponse,
    MarketDataResponse,
    RiskMetricsResponse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_market_data(
    ticker: str = "AAPL",
    current_price: float = 150.0,
    price_change_pct: float = 2.5,
) -> MarketData:
    return MarketData(
        ticker=ticker,
        current_price=current_price,
        price_change_pct=price_change_pct,
        volume=50_000_000.0,
        market_cap=2_500_000_000_000.0,
        pe_ratio=28.5,
        week_52_high=198.0,
        week_52_low=124.0,
        avg_volume=55_000_000.0,
    )


def make_price_history(n: int = 100, trend: float = 0.001) -> list:
    """Generate synthetic price history."""
    np.random.seed(42)
    prices = 100.0 * np.cumprod(1 + np.random.normal(trend, 0.015, n))
    return [
        {
            "date": f"2024-{(i // 20) + 1:02d}-{(i % 20) + 1:02d}",
            "open": float(prices[i] * 0.999),
            "high": float(prices[i] * 1.005),
            "low": float(prices[i] * 0.995),
            "close": float(prices[i]),
            "volume": 1_000_000,
        }
        for i in range(n)
    ]


def make_backtest_results(
    sharpe: float = 1.2,
    max_dd: float = 0.12,
    var_95: float = 0.02,
    win_rate: float = 0.55,
    total_return: float = 0.25,
    n_trades: int = 20,
) -> BacktestResults:
    return BacktestResults(
        strategy_name="Momentum_SMA_20_50",
        total_return=total_return,
        annualized_return=0.12,
        sharpe_ratio=sharpe,
        sortino_ratio=1.5,
        max_drawdown=max_dd,
        calmar_ratio=1.0,
        var_95=var_95,
        cvar_95=var_95 * 1.3,
        win_rate=win_rate,
        profit_factor=1.8,
        n_trades=n_trades,
        n_days=252,
        start_date="2022-01-01",
        end_date="2024-12-31",
    )


def make_state(
    ticker: str = "AAPL",
    query: str = "Should I buy AAPL?",
    retry_count: int = 0,
    selected_strategy: str = "momentum",
    backtest: BacktestResults | None = None,
) -> TradingState:
    state = create_initial_state(ticker, query)
    state["retry_count"] = retry_count
    state["selected_strategy"] = selected_strategy
    state["market_data"] = make_market_data(ticker)
    state["price_history"] = make_price_history()
    state["rag_context"] = "Apple revenue grew 8% in Q3 2024."
    state["strategy_params"] = {"short_window": 20, "long_window": 50}
    if backtest:
        state["backtest_results"] = backtest
    return state


# ---------------------------------------------------------------------------
# TradingState tests
# ---------------------------------------------------------------------------

class TestTradingState:
    def test_create_initial_state_ticker_uppercased(self) -> None:
        state = create_initial_state("aapl", "test query")
        assert state["ticker"] == "AAPL"

    def test_create_initial_state_defaults(self) -> None:
        state = create_initial_state("MSFT", "test")
        assert state["retry_count"] == 0
        assert state["risk_approved"] is False
        assert state["signal"] == "HOLD"
        assert state["error"] == ""
        assert state["citations"] == []

    def test_create_initial_state_custom_dates(self) -> None:
        state = create_initial_state("AAPL", "test", "2023-01-01", "2023-12-31")
        assert state["start_date"] == "2023-01-01"
        assert state["end_date"] == "2023-12-31"

    def test_state_is_dict_like(self) -> None:
        state = create_initial_state("AAPL", "test")
        assert "ticker" in state
        assert state.get("ticker") == "AAPL"


# ---------------------------------------------------------------------------
# StrategyAgent tests
# ---------------------------------------------------------------------------

class TestStrategyAgent:
    def test_select_momentum_returns_correct_strategy(self) -> None:
        state = make_state()
        result = _select_momentum(state, "Test rationale")
        assert result["selected_strategy"] == "momentum"
        assert result["strategy_params"]["short_window"] == 20
        assert result["strategy_params"]["long_window"] == 50

    def test_select_mean_reversion_returns_correct_strategy(self) -> None:
        state = make_state()
        result = _select_mean_reversion(state, "Test rationale", volatility=0.02)
        assert result["selected_strategy"] == "mean_reversion"
        assert "window" in result["strategy_params"]
        assert "z_threshold" in result["strategy_params"]

    def test_mean_reversion_high_volatility_wider_threshold(self) -> None:
        state = make_state()
        result = _select_mean_reversion(state, "High vol", volatility=0.03)
        assert result["strategy_params"]["z_threshold"] == 2.5

    def test_mean_reversion_low_volatility_tighter_threshold(self) -> None:
        state = make_state()
        result = _select_mean_reversion(state, "Low vol", volatility=0.01)
        assert result["strategy_params"]["z_threshold"] == 1.5

    def test_strategy_agent_trending_market_selects_momentum(self) -> None:
        state = make_state()
        # Strong uptrend: 10% price change
        state["market_data"] = make_market_data(price_change_pct=10.0)
        state["price_history"] = make_price_history(trend=0.005)  # Strong trend

        result = asyncio.get_event_loop().run_until_complete(strategy_agent(state))
        assert result["selected_strategy"] == "momentum"

    def test_strategy_agent_retry_switches_strategy(self) -> None:
        state = make_state(retry_count=1, selected_strategy="momentum")
        result = asyncio.get_event_loop().run_until_complete(strategy_agent(state))
        # On retry from momentum → should switch to mean_reversion
        assert result["selected_strategy"] == "mean_reversion"

    def test_strategy_agent_retry_from_mean_reversion_switches_to_momentum(self) -> None:
        state = make_state(retry_count=1, selected_strategy="mean_reversion")
        result = asyncio.get_event_loop().run_until_complete(strategy_agent(state))
        assert result["selected_strategy"] == "momentum"

    def test_strategy_agent_insufficient_history_defaults_to_momentum(self) -> None:
        state = make_state()
        state["price_history"] = make_price_history(n=10)  # Too short
        result = asyncio.get_event_loop().run_until_complete(strategy_agent(state))
        assert result["selected_strategy"] == "momentum"

    def test_strategy_agent_sets_rationale(self) -> None:
        state = make_state()
        result = asyncio.get_event_loop().run_until_complete(strategy_agent(state))
        assert len(result.get("strategy_rationale", "")) > 0


# ---------------------------------------------------------------------------
# RiskAgent tests
# ---------------------------------------------------------------------------

class TestRiskAgent:
    def test_good_strategy_approved(self) -> None:
        state = make_state(backtest=make_backtest_results(
            sharpe=1.5, max_dd=0.10, var_95=0.02, win_rate=0.55
        ))
        result = asyncio.get_event_loop().run_until_complete(risk_agent(state))
        assert result["risk_approved"] is True

    def test_low_sharpe_rejected(self) -> None:
        state = make_state(backtest=make_backtest_results(sharpe=0.2))
        result = asyncio.get_event_loop().run_until_complete(risk_agent(state))
        assert result["risk_approved"] is False
        assert result["retry_count"] == 1

    def test_high_drawdown_rejected(self) -> None:
        state = make_state(backtest=make_backtest_results(max_dd=0.35))
        result = asyncio.get_event_loop().run_until_complete(risk_agent(state))
        assert result["risk_approved"] is False

    def test_high_var_rejected(self) -> None:
        state = make_state(backtest=make_backtest_results(var_95=0.08))
        result = asyncio.get_event_loop().run_until_complete(risk_agent(state))
        assert result["risk_approved"] is False

    def test_max_retries_approves_with_warning(self) -> None:
        state = make_state(
            retry_count=3,
            backtest=make_backtest_results(sharpe=0.1),  # Bad strategy
        )
        result = asyncio.get_event_loop().run_until_complete(risk_agent(state))
        # After max retries, should approve anyway
        assert result["risk_approved"] is True
        assert "retries" in result["risk_metrics"]["rejection_reason"].lower()

    def test_retry_count_increments_on_rejection(self) -> None:
        state = make_state(retry_count=0, backtest=make_backtest_results(sharpe=0.1))
        result = asyncio.get_event_loop().run_until_complete(risk_agent(state))
        assert result["retry_count"] == 1

    def test_risk_metrics_populated(self) -> None:
        state = make_state(backtest=make_backtest_results())
        result = asyncio.get_event_loop().run_until_complete(risk_agent(state))
        rm = result["risk_metrics"]
        assert "risk_score" in rm
        assert "risk_level" in rm
        assert rm["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_should_retry_approved(self) -> None:
        state = make_state()
        state["risk_approved"] = True
        assert should_retry(state) == "approved"

    def test_should_retry_rejected(self) -> None:
        state = make_state()
        state["risk_approved"] = False
        assert should_retry(state) == "retry"

    def test_compute_risk_score_good_strategy(self) -> None:
        score = _compute_risk_score(sharpe=2.0, max_dd=0.05, var_95=0.01, win_rate=0.6)
        assert score < 3.0  # Should be LOW risk

    def test_compute_risk_score_bad_strategy(self) -> None:
        score = _compute_risk_score(sharpe=0.0, max_dd=0.30, var_95=0.06, win_rate=0.3)
        assert score > 6.0  # Should be HIGH risk

    def test_get_risk_level_low(self) -> None:
        assert _get_risk_level(2.0) == "LOW"

    def test_get_risk_level_medium(self) -> None:
        assert _get_risk_level(4.5) == "MEDIUM"

    def test_get_risk_level_high(self) -> None:
        assert _get_risk_level(7.0) == "HIGH"


# ---------------------------------------------------------------------------
# ExplainerAgent tests
# ---------------------------------------------------------------------------

class TestExplainerAgent:
    def test_parse_signal_buy(self) -> None:
        assert _parse_signal("SIGNAL: BUY\nApple looks strong.") == "BUY"

    def test_parse_signal_sell(self) -> None:
        assert _parse_signal("SIGNAL: SELL\nRisk too high.") == "SELL"

    def test_parse_signal_hold(self) -> None:
        assert _parse_signal("SIGNAL: HOLD\nWait for clarity.") == "HOLD"

    def test_parse_signal_fallback_count(self) -> None:
        # More BUY mentions → BUY
        assert _parse_signal("BUY BUY BUY SELL HOLD") == "BUY"

    def test_parse_signal_default_hold(self) -> None:
        assert _parse_signal("No clear signal here.") == "HOLD"

    def test_signal_from_backtest_buy(self) -> None:
        state = make_state(backtest=make_backtest_results(sharpe=1.5, total_return=0.20))
        assert _signal_from_backtest(state) == "BUY"

    def test_signal_from_backtest_sell(self) -> None:
        state = make_state(backtest=make_backtest_results(sharpe=-0.5, total_return=-0.15))
        assert _signal_from_backtest(state) == "SELL"

    def test_signal_from_backtest_hold(self) -> None:
        state = make_state(backtest=make_backtest_results(sharpe=0.3, total_return=0.02))
        assert _signal_from_backtest(state) == "HOLD"

    def test_fallback_explanation_contains_ticker(self) -> None:
        state = make_state(backtest=make_backtest_results())
        explanation = _generate_fallback_explanation(state)
        assert "AAPL" in explanation

    def test_fallback_explanation_contains_signal(self) -> None:
        state = make_state(backtest=make_backtest_results(sharpe=1.5, total_return=0.20))
        explanation = _generate_fallback_explanation(state)
        assert "SIGNAL:" in explanation

    def test_fallback_explanation_contains_metrics(self) -> None:
        state = make_state(backtest=make_backtest_results(sharpe=1.42))
        explanation = _generate_fallback_explanation(state)
        assert "1.42" in explanation  # Sharpe ratio

    def test_fallback_explanation_mentions_groq_key(self) -> None:
        state = make_state(backtest=make_backtest_results())
        explanation = _generate_fallback_explanation(state)
        assert "GROQ_API_KEY" in explanation


# ---------------------------------------------------------------------------
# API Schema tests
# ---------------------------------------------------------------------------

class TestAnalysisRequest:
    def test_valid_request(self) -> None:
        req = AnalysisRequest(
            ticker="aapl",
            query="Should I buy Apple?",
            start_date="2022-01-01",
            end_date="2024-12-31",
        )
        assert req.ticker == "AAPL"  # Uppercased

    def test_empty_ticker_raises(self) -> None:
        with pytest.raises(Exception):
            AnalysisRequest(ticker="", query="test")

    def test_invalid_date_format_raises(self) -> None:
        with pytest.raises(Exception):
            AnalysisRequest(ticker="AAPL", query="test", start_date="01/01/2022")

    def test_empty_query_raises(self) -> None:
        with pytest.raises(Exception):
            AnalysisRequest(ticker="AAPL", query="   ")

    def test_query_stripped(self) -> None:
        req = AnalysisRequest(ticker="AAPL", query="  Should I buy?  ")
        assert req.query == "Should I buy?"

    def test_default_dates(self) -> None:
        req = AnalysisRequest(ticker="AAPL", query="test query here")
        assert req.start_date == "2022-01-01"
        assert req.end_date == "2024-12-31"


class TestAnalysisResponse:
    def test_from_trading_state_basic(self) -> None:
        state = make_state(backtest=make_backtest_results())
        state["signal"] = "BUY"
        state["final_explanation"] = "SIGNAL: BUY\nApple looks strong."
        state["final_citations"] = ["[SEC EDGAR] Apple 10-K 2024"]
        state["risk_metrics"] = RiskMetrics(
            sharpe_ratio=1.2,
            max_drawdown=0.12,
            var_95=0.02,
            risk_score=3.5,
            risk_level="MEDIUM",
            risk_approved=True,
            rejection_reason="",
        )
        state["processing_time_ms"] = 5000.0

        response = AnalysisResponse.from_trading_state(state)
        assert response.ticker == "AAPL"
        assert response.signal == "BUY"
        assert response.processing_time_ms == 5000.0

    def test_from_trading_state_with_market_data(self) -> None:
        state = make_state()
        state["signal"] = "HOLD"
        state["final_explanation"] = "HOLD for now."
        response = AnalysisResponse.from_trading_state(state)
        assert response.market_data is not None
        assert response.market_data.ticker == "AAPL"
        assert response.market_data.current_price == 150.0

    def test_from_trading_state_empty_state(self) -> None:
        state = create_initial_state("MSFT", "test")
        response = AnalysisResponse.from_trading_state(state)
        assert response.ticker == "MSFT"
        assert response.signal == "HOLD"
        assert response.backtest_results is None
        assert response.market_data is None

    def test_equity_curve_limited_to_252(self) -> None:
        state = create_initial_state("AAPL", "test")
        state["equity_curve"] = [100.0 + i for i in range(500)]
        state["signal"] = "HOLD"
        state["final_explanation"] = "test"
        response = AnalysisResponse.from_trading_state(state)
        assert len(response.equity_curve) <= 252
