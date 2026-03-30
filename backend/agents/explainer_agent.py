"""ExplainerAgent — generates a cited, explainable trading recommendation.

This is the FINAL agent in the LangGraph workflow. It:
1. Combines all previous agent outputs into a structured prompt
2. Calls Groq LLM (Llama 3.1 70B — free) to generate the explanation
3. Produces a BUY/SELL/HOLD signal with confidence
4. Returns a cited answer with source references

Why Groq (free tier):
    - 14,400 requests/day free
    - Llama 3.1 70B: excellent reasoning quality
    - ~500ms response time (fast enough for API)
    - No credit card required

Prompt engineering:
    The prompt is structured as a financial analyst briefing:
    1. Market context (from ResearchAgent)
    2. Fundamental analysis (from RAGAgent — SEC/news)
    3. Strategy selection rationale (from StrategyAgent)
    4. Backtest performance (from BacktestAgent)
    5. Risk assessment (from RiskAgent)
    → LLM synthesizes all into a cited recommendation

LangGraph node contract:
    Input:  TradingState with all previous agent outputs
    Output: TradingState with final_explanation, final_citations, signal
"""

from __future__ import annotations

from config.logging_config import get_logger
from config.settings import settings
from graph.state import TradingState

logger = get_logger(__name__)

# System prompt for the financial analyst role
SYSTEM_PROMPT = """You are QuantMind, an expert AI trading strategy advisor.
You analyze stocks using quantitative backtesting results and fundamental analysis
from SEC filings and financial news.

Your responses must:
1. Be concise and actionable (200-300 words)
2. Start with a clear BUY / SELL / HOLD signal
3. Cite specific sources using [Source N] notation
4. Explain the quantitative evidence (Sharpe ratio, drawdown)
5. Explain the fundamental evidence (from SEC filings/news)
6. State the key risk factors
7. End with a confidence level: HIGH / MEDIUM / LOW

Format:
SIGNAL: [BUY/SELL/HOLD]
CONFIDENCE: [HIGH/MEDIUM/LOW]

[Your 200-300 word explanation with citations]

SOURCES:
[List all cited sources]"""


async def explainer_agent(state: TradingState) -> TradingState:
    """Generate cited trading recommendation. LangGraph node function.

    Synthesizes all agent outputs into a human-readable, cited explanation
    using Groq's Llama 3.1 70B model (free tier).

    Args:
        state: TradingState with all previous agent outputs populated.

    Returns:
        Updated TradingState with final_explanation, final_citations, signal.

    Example:
        >>> state = await explainer_agent(state)
        >>> print(state["signal"])
        'BUY'
        >>> print(state["final_explanation"][:100])
        'SIGNAL: BUY\nCONFIDENCE: MEDIUM\n\nApple shows a golden cross...'
    """
    ticker = state.get("ticker", "")
    query = state.get("query", f"Should I trade {ticker}?")

    logger.info("ExplainerAgent | generating | ticker=%s", ticker)

    try:
        # Build the analysis prompt
        prompt = _build_prompt(state)

        # Call Groq LLM
        explanation = await _call_groq(prompt)

        # Parse signal from response
        signal = _parse_signal(explanation)

        # Combine RAG citations with final answer
        rag_citations = state.get("citations", [])
        final_citations = rag_citations[:5]  # Top 5 citations

        logger.info(
            "ExplainerAgent | complete | ticker=%s | signal=%s | chars=%d",
            ticker, signal, len(explanation),
        )

        return {
            **state,
            "final_explanation": explanation,
            "final_citations": final_citations,
            "signal": signal,
        }

    except Exception as e:
        logger.error(
            "ExplainerAgent | failed | ticker=%s | %s", ticker, e, exc_info=True
        )
        # Fallback: generate explanation from backtest data without LLM
        fallback = _generate_fallback_explanation(state)
        return {
            **state,
            "final_explanation": fallback,
            "final_citations": state.get("citations", [])[:3],
            "signal": _signal_from_backtest(state),
            "error": f"ExplainerAgent LLM failed: {e}. Using fallback.",
        }


def _build_prompt(state: TradingState) -> str:
    """Build the analysis prompt from all agent outputs.

    Args:
        state: Full TradingState.

    Returns:
        Formatted prompt string for the LLM.
    """
    ticker = state.get("ticker", "")
    query = state.get("query", "")
    market_data = state.get("market_data", {})
    rag_context = state.get("rag_context", "No documents available.")
    strategy_rationale = state.get("strategy_rationale", "")
    selected_strategy = state.get("selected_strategy", "momentum")
    backtest = state.get("backtest_results", {})
    risk = state.get("risk_metrics", {})

    # Format market data
    price = market_data.get("current_price", 0)
    change = market_data.get("price_change_pct", 0)
    high_52 = market_data.get("week_52_high", 0)
    low_52 = market_data.get("week_52_low", 0)

    # Format backtest metrics
    sharpe = backtest.get("sharpe_ratio", 0)
    total_ret = backtest.get("total_return", 0)
    max_dd = backtest.get("max_drawdown", 0)
    win_rate = backtest.get("win_rate", 0)
    n_trades = backtest.get("n_trades", 0)
    start = backtest.get("start_date", "")
    end = backtest.get("end_date", "")

    # Format risk metrics
    risk_level = risk.get("risk_level", "UNKNOWN")
    risk_score = risk.get("risk_score", 0)

    prompt = f"""USER QUERY: {query}

TICKER: {ticker}

=== MARKET DATA ===
Current Price: ${price:.2f} ({change:+.2f}% today)
52-Week Range: ${low_52:.2f} — ${high_52:.2f}

=== FUNDAMENTAL ANALYSIS (from SEC filings & news) ===
{rag_context}

=== STRATEGY SELECTION ===
Selected Strategy: {selected_strategy.upper()}
Rationale: {strategy_rationale}

=== BACKTEST RESULTS ({start} to {end}) ===
Total Return:      {total_ret:.1%}
Sharpe Ratio:      {sharpe:.2f}
Max Drawdown:      {max_dd:.1%}
Win Rate:          {win_rate:.1%}
Number of Trades:  {n_trades}

=== RISK ASSESSMENT ===
Risk Level: {risk_level}
Risk Score: {risk_score:.1f}/10

Based on all the above analysis, provide your trading recommendation for {ticker}.
Answer the user's question: "{query}"
"""
    return prompt


async def _call_groq(prompt: str) -> str:
    """Call Groq API with the analysis prompt.

    Uses Groq's free tier (14,400 req/day) with Llama 3.1 70B.

    Args:
        prompt: Formatted analysis prompt.

    Returns:
        LLM response string.

    Raises:
        Exception: If Groq API call fails.
    """
    if not settings.groq_api_key:
        logger.warning("ExplainerAgent | no Groq API key — using fallback")
        raise ValueError("GROQ_API_KEY not configured. Add it to .env file.")

    try:
        from groq import Groq  # type: ignore[import]

        client = Groq(api_key=settings.groq_api_key)

        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.3,  # Low temperature for consistent, factual responses
        )

        return response.choices[0].message.content or ""

    except ImportError:
        raise ImportError(
            "groq package not installed. Run: pip install groq"
        )


def _parse_signal(explanation: str) -> str:
    """Extract BUY/SELL/HOLD signal from LLM response.

    Args:
        explanation: LLM response text.

    Returns:
        Signal string: 'BUY', 'SELL', or 'HOLD'.
    """
    upper = explanation.upper()
    if "SIGNAL: BUY" in upper or upper.startswith("BUY"):
        return "BUY"
    elif "SIGNAL: SELL" in upper or upper.startswith("SELL"):
        return "SELL"
    elif "SIGNAL: HOLD" in upper or upper.startswith("HOLD"):
        return "HOLD"

    # Fallback: count mentions
    buy_count = upper.count("BUY")
    sell_count = upper.count("SELL")
    hold_count = upper.count("HOLD")

    if buy_count > sell_count and buy_count > hold_count:
        return "BUY"
    elif sell_count > buy_count and sell_count > hold_count:
        return "SELL"
    return "HOLD"


def _signal_from_backtest(state: TradingState) -> str:
    """Determine signal from backtest results (fallback when LLM fails).

    Args:
        state: TradingState with backtest_results.

    Returns:
        Signal string based on Sharpe ratio and total return.
    """
    backtest = state.get("backtest_results", {})
    sharpe = backtest.get("sharpe_ratio", 0.0)
    total_return = backtest.get("total_return", 0.0)

    if sharpe > 1.0 and total_return > 0.05:
        return "BUY"
    elif sharpe < 0.0 or total_return < -0.10:
        return "SELL"
    return "HOLD"


def _generate_fallback_explanation(state: TradingState) -> str:
    """Generate a rule-based explanation when LLM is unavailable.

    Args:
        state: Full TradingState.

    Returns:
        Formatted explanation string without LLM.
    """
    ticker = state.get("ticker", "")
    backtest = state.get("backtest_results", {})
    risk = state.get("risk_metrics", {})
    strategy = state.get("selected_strategy", "momentum")
    citations = state.get("citations", [])

    signal = _signal_from_backtest(state)
    sharpe = backtest.get("sharpe_ratio", 0.0)
    total_ret = backtest.get("total_return", 0.0)
    max_dd = backtest.get("max_drawdown", 0.0)
    risk_level = risk.get("risk_level", "UNKNOWN")

    citation_text = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(citations[:3]))

    return f"""SIGNAL: {signal}
CONFIDENCE: MEDIUM

{ticker} analysis using {strategy.upper()} strategy:

Quantitative Evidence:
- Total Return: {total_ret:.1%} over backtest period
- Sharpe Ratio: {sharpe:.2f} (risk-adjusted performance)
- Maximum Drawdown: {max_dd:.1%}
- Risk Level: {risk_level}

The {strategy} strategy {'shows positive momentum' if signal == 'BUY' else 'indicates caution'} for {ticker}.
{'Strong risk-adjusted returns support a BUY recommendation.' if signal == 'BUY' else
 'Risk metrics suggest caution — consider HOLD or reduced position.' if signal == 'HOLD' else
 'Negative performance metrics suggest avoiding this position.'}

Note: LLM explanation unavailable. Configure GROQ_API_KEY in .env for AI-powered analysis.

SOURCES:
{citation_text if citation_text else 'No sources available.'}"""
