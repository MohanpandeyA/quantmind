"""LangGraph workflow — the state machine connecting all QuantMind agents.

Defines the directed graph of agent nodes and edges:

    START
      ↓
    research_agent    (fetch market data from yfinance)
      ↓
    rag_agent         (retrieve SEC/news documents from ChromaDB)
      ↓
    strategy_agent    (select Momentum or MeanReversion)
      ↓
    backtest_agent    (run historical backtest)
      ↓
    risk_agent        (evaluate risk limits)
      ↓ (conditional)
    ┌─ approved → explainer_agent → END
    └─ rejected → strategy_agent (retry, max 3 times)

Key LangGraph concepts used:
    - StateGraph: the graph container with typed state
    - add_node(): register each agent as a node
    - add_edge(): connect nodes sequentially
    - add_conditional_edges(): branch based on state value
    - compile(): produce the runnable graph

Why LangGraph over plain async:
    - Built-in state management (no manual state passing)
    - Conditional routing (retry logic is declarative)
    - Checkpointing (can resume interrupted workflows)
    - Streaming (can stream partial results to frontend)
    - Visualization (can render the graph as a diagram)
"""

from __future__ import annotations

import time
from typing import Any

from config.logging_config import get_logger
from graph.state import TradingState, create_initial_state

logger = get_logger(__name__)


def build_workflow() -> Any:
    """Build and compile the LangGraph trading advisor workflow.

    Returns:
        Compiled LangGraph StateGraph ready to invoke.

    Raises:
        ImportError: If langgraph is not installed.

    Example:
        >>> app = build_workflow()
        >>> result = await app.ainvoke(state)
    """
    try:
        from langgraph.graph import StateGraph, END  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "langgraph not installed. Run: pip install langgraph"
        )

    from agents.backtest_agent import backtest_agent
    from agents.explainer_agent import explainer_agent
    from agents.rag_agent import rag_agent
    from agents.research_agent import research_agent
    from agents.risk_agent import risk_agent, should_retry
    from agents.strategy_agent import strategy_agent

    # Create the state graph with TradingState as the shared state type
    workflow = StateGraph(TradingState)

    # --- Register all agent nodes ---
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("strategy_agent", strategy_agent)
    workflow.add_node("backtest_agent", backtest_agent)
    workflow.add_node("risk_agent", risk_agent)
    workflow.add_node("explainer_agent", explainer_agent)

    # --- Define sequential edges ---
    workflow.set_entry_point("research_agent")
    workflow.add_edge("research_agent", "rag_agent")
    workflow.add_edge("rag_agent", "strategy_agent")
    workflow.add_edge("strategy_agent", "backtest_agent")
    workflow.add_edge("backtest_agent", "risk_agent")

    # --- Conditional edge: risk_agent → retry or proceed ---
    # should_retry() returns 'approved' or 'retry'
    workflow.add_conditional_edges(
        "risk_agent",
        should_retry,
        {
            "approved": "explainer_agent",  # Risk OK → generate explanation
            "retry": "strategy_agent",       # Risk rejected → try different strategy
        },
    )

    # --- Final edge ---
    workflow.add_edge("explainer_agent", END)

    # Compile the graph into a runnable
    app = workflow.compile()

    logger.info("LangGraph workflow compiled | nodes=6 | conditional_edges=1")
    return app


# Singleton compiled workflow — built once, reused across requests
_workflow = None


def get_workflow() -> Any:
    """Get or create the singleton compiled workflow.

    Returns:
        Compiled LangGraph workflow.
    """
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


async def run_analysis(
    ticker: str,
    query: str,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
) -> TradingState:
    """Run the full trading analysis workflow.

    This is the main entry point for the FastAPI endpoint.
    Creates initial state, runs all agents, returns final state.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').
        query: User's natural language question.
        start_date: Backtest start date. Default '2022-01-01'.
        end_date: Backtest end date. Default '2024-12-31'.

    Returns:
        Final TradingState with all agent outputs populated.

    Example:
        >>> result = await run_analysis("AAPL", "Should I buy AAPL?")
        >>> print(result["signal"])
        'BUY'
        >>> print(result["final_explanation"][:100])
        'SIGNAL: BUY\nCONFIDENCE: MEDIUM\n\nApple shows...'
    """
    start_time = time.perf_counter()

    # Create initial state
    initial_state = create_initial_state(
        ticker=ticker,
        query=query,
        start_date=start_date,
        end_date=end_date,
    )

    logger.info(
        "Workflow | starting | ticker=%s | query=%r",
        ticker, query[:50],
    )

    try:
        app = get_workflow()

        # Run the full workflow asynchronously
        final_state: TradingState = await app.ainvoke(initial_state)

        # Record total processing time
        duration_ms = (time.perf_counter() - start_time) * 1000
        final_state["processing_time_ms"] = round(duration_ms, 1)

        logger.info(
            "Workflow | complete | ticker=%s | signal=%s | time=%.0fms",
            ticker,
            final_state.get("signal", "HOLD"),
            duration_ms,
        )

        return final_state

    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "Workflow | failed | ticker=%s | %s | time=%.0fms",
            ticker, e, duration_ms, exc_info=True,
        )
        return {
            **initial_state,
            "error": str(e),
            "signal": "HOLD",
            "final_explanation": f"Analysis failed: {e}",
            "processing_time_ms": round(duration_ms, 1),
        }
