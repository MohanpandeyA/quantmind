"""LangGraph shared state schema for the QuantMind trading advisor.

TradingState is the single shared data structure that flows through
all agents in the LangGraph workflow. Each agent reads from and writes
to this state — no direct agent-to-agent communication.

Why TypedDict for state:
    - LangGraph requires TypedDict for state definitions
    - Type-safe: mypy catches missing fields at development time
    - Serializable: can be persisted to MongoDB for session history
    - Immutable updates: LangGraph creates new state copies, never mutates

State flow through agents:
    START
    → ResearchAgent:  fills market_data, price_history
    → RAGAgent:       fills retrieved_docs, rag_context, citations
    → StrategyAgent:  fills selected_strategy, strategy_params
    → BacktestAgent:  fills backtest_results, equity_curve
    → RiskAgent:      fills risk_metrics, risk_approved (or retry)
    → ExplainerAgent: fills final_explanation, final_citations
    END

Retry logic:
    RiskAgent can send state back to StrategyAgent (max 3 retries)
    if risk limits are breached. retry_count tracks this.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class MarketData(TypedDict, total=False):
    """Market data fetched by ResearchAgent.

    Attributes:
        ticker: Stock ticker symbol.
        current_price: Latest closing price.
        price_change_pct: % change from previous close.
        volume: Latest trading volume.
        market_cap: Market capitalization.
        pe_ratio: Price-to-earnings ratio.
        week_52_high: 52-week high price.
        week_52_low: 52-week low price.
        avg_volume: 30-day average volume.
    """

    ticker: str
    current_price: float
    price_change_pct: float
    volume: float
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    week_52_high: float
    week_52_low: float
    avg_volume: float


class BacktestResults(TypedDict, total=False):
    """Backtest results from BacktestAgent.

    Attributes:
        strategy_name: Name of the strategy tested.
        total_return: Total return over the period.
        annualized_return: Annualized return.
        sharpe_ratio: Risk-adjusted return.
        sortino_ratio: Downside risk-adjusted return.
        max_drawdown: Maximum peak-to-trough loss.
        calmar_ratio: Annualized return / max drawdown.
        var_95: Value at Risk at 95% confidence.
        cvar_95: Conditional VaR (Expected Shortfall).
        win_rate: Fraction of profitable trades.
        profit_factor: Gross profit / gross loss.
        n_trades: Total number of trades.
        n_days: Total trading days in backtest.
        start_date: Backtest start date.
        end_date: Backtest end date.
    """

    strategy_name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    win_rate: float
    profit_factor: float
    n_trades: int
    n_days: int
    start_date: str
    end_date: str


class RiskMetrics(TypedDict, total=False):
    """Risk assessment from RiskAgent.

    Attributes:
        sharpe_ratio: Strategy Sharpe ratio.
        max_drawdown: Maximum drawdown.
        var_95: Value at Risk.
        risk_score: Composite risk score (0-10, lower = safer).
        risk_level: Human-readable risk level (LOW/MEDIUM/HIGH).
        risk_approved: Whether risk limits are satisfied.
        rejection_reason: Why risk was rejected (if applicable).
    """

    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    risk_score: float
    risk_level: str
    risk_approved: bool
    rejection_reason: str


class TradingState(TypedDict, total=False):
    """Shared state flowing through all LangGraph agents.

    This is the single source of truth for the entire agent workflow.
    Each agent reads relevant fields and writes its outputs back.

    Fields marked Optional are populated by specific agents:
        - market_data:        ResearchAgent
        - price_history:      ResearchAgent
        - retrieved_docs:     RAGAgent
        - rag_context:        RAGAgent
        - citations:          RAGAgent
        - selected_strategy:  StrategyAgent
        - strategy_params:    StrategyAgent
        - strategy_rationale: StrategyAgent
        - backtest_results:   BacktestAgent
        - equity_curve:       BacktestAgent
        - risk_metrics:       RiskAgent
        - risk_approved:      RiskAgent
        - final_explanation:  ExplainerAgent
        - final_citations:    ExplainerAgent
        - signal:             ExplainerAgent (BUY/SELL/HOLD)

    Attributes:
        ticker: Stock ticker symbol (e.g., 'AAPL').
        query: User's natural language question.
        start_date: Backtest start date (YYYY-MM-DD).
        end_date: Backtest end date (YYYY-MM-DD).
        market_data: Current market data dict.
        price_history: List of OHLCV dicts for charting.
        retrieved_docs: Raw retrieved document chunks.
        rag_context: Formatted context string for LLM.
        citations: Source citation strings.
        selected_strategy: Strategy name ('momentum' or 'mean_reversion').
        strategy_params: Strategy hyperparameters dict.
        strategy_rationale: Why this strategy was selected.
        backtest_results: Full backtest performance metrics.
        equity_curve: List of equity values for charting.
        risk_metrics: Risk assessment results.
        risk_approved: Whether risk limits are satisfied.
        retry_count: Number of strategy retry attempts (max 3).
        final_explanation: LLM-generated cited explanation.
        final_citations: Final source citations list.
        signal: Trading signal (BUY/SELL/HOLD).
        error: Error message if any agent failed.
        processing_time_ms: Total processing time in milliseconds.
    """

    # --- Input fields (set by API caller) ---
    ticker: str
    query: str
    start_date: str
    end_date: str

    # --- ResearchAgent outputs ---
    market_data: MarketData
    price_history: List[Dict[str, Any]]

    # --- RAGAgent outputs ---
    retrieved_docs: List[str]
    rag_context: str
    citations: List[str]

    # --- StrategyAgent outputs ---
    selected_strategy: str
    strategy_params: Dict[str, Any]
    strategy_rationale: str

    # --- BacktestAgent outputs ---
    backtest_results: BacktestResults
    equity_curve: List[float]

    # --- RiskAgent outputs ---
    risk_metrics: RiskMetrics
    risk_approved: bool
    retry_count: int

    # --- ExplainerAgent outputs ---
    final_explanation: str
    final_citations: List[str]
    signal: str

    # --- Meta ---
    error: str
    processing_time_ms: float


def create_initial_state(
    ticker: str,
    query: str,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
) -> TradingState:
    """Create an initial TradingState with required fields.

    Args:
        ticker: Stock ticker symbol.
        query: User's natural language question.
        start_date: Backtest start date. Default '2022-01-01'.
        end_date: Backtest end date. Default '2024-12-31'.

    Returns:
        TradingState with required fields populated and defaults set.

    Example:
        >>> state = create_initial_state("AAPL", "Should I buy AAPL?")
        >>> state["ticker"]
        'AAPL'
        >>> state["retry_count"]
        0
    """
    return TradingState(
        ticker=ticker.upper(),
        query=query,
        start_date=start_date,
        end_date=end_date,
        retry_count=0,
        risk_approved=False,
        signal="HOLD",
        error="",
        citations=[],
        final_citations=[],
        retrieved_docs=[],
        equity_curve=[],
        price_history=[],
    )
