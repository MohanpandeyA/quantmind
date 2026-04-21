# QuantMind — AI-Powered Algorithmic Trading Strategy Advisor

> **LangGraph + RAG + FinBERT + MERN + DSA** | **453 Tests Passing ✅** | **8 Dashboard Tabs** | 100% Free to Run

A production-grade fintech system that combines **quantitative trading strategies**, **AI-powered document retrieval**, **real-time sentiment analysis**, and **explainable recommendations** — built with professional code practices across a full-stack architecture.

---

## 🎯 What Problem It Solves

Most retail traders and small hedge funds face:
1. **Strategy Overload** — thousands of strategies exist, no intelligent selection
2. **Black-box AI** — existing tools give signals without explanation
3. **Knowledge Fragmentation** — financial knowledge scattered across SEC filings, news, earnings
4. **Backtesting Blindness** — traders backtest without understanding *why* a strategy worked
5. **Overfitting Risk** — strategies that look great in backtests fail in live trading

**QuantMind solves all five** with a 7-agent LangGraph pipeline that backtests strategies, retrieves real financial documents, scores sentiment with FinBERT, validates robustness with walk-forward testing, and generates cited AI explanations.

---

## 🏗️ System Architecture

```
User Query: "Should I buy AAPL?"
         ↓
React Dashboard (port 5173) — 8 tabs
         ↓ POST /analyze
FastAPI (port 8000)
         ↓ runs LangGraph workflow
ResearchAgent   → yfinance: fetch AAPL price data (free, no key)
         ↓
RAGAgent        → SEC 10-K + News → ChromaDB → MMR retrieval
         ↓
SentimentAgent  → FinBERT (ProsusAI/finbert) → BULLISH/BEARISH/NEUTRAL
         ↓
StrategyAgent   → Select: Momentum / MeanReversion / RSI / MACD
                  (sentiment biases selection: bearish → MeanReversion)
         ↓
BacktestAgent   → Run on 3 years of real data → Sharpe=1.4, MDD=12%
         ↓
RiskAgent       → VaR=2.1%, CVaR=3.4% → Risk approved (or retry ×3)
         ↓
ExplainerAgent  → "BUY signal. iPhone revenue grew 8% (Apple 10-K 2024).
                   Services hit record $24B. Momentum confirmed."
         ↓
React Dashboard shows:
  ├── 📈 BUY/SELL/HOLD signal badge
  ├── 📊 Equity curve chart (Recharts AreaChart)
  ├── 📋 Metrics table (Sharpe, VaR, Drawdown, Win Rate)
  ├── 🧠 FinBERT sentiment score + top sentences
  └── 🤖 AI explanation with source citations
```

---

## 📦 Tech Stack

| Layer | Technology | Cost |
|-------|-----------|------|
| AI Orchestration | LangGraph (7-agent state machine) | Free |
| RAG Pipeline | ChromaDB + sentence-transformers (all-MiniLM-L6-v2) | Free (local) |
| Sentiment Analysis | FinBERT (ProsusAI/finbert) | Free (local) |
| Market Data | yfinance + SEC EDGAR API | Free (no key) |
| DSA Engine | Segment Tree O(log n) + Welford's O(1) | — |
| Trading Strategies | Momentum, MeanReversion, RSI, MACD | — |
| Walk-Forward Validation | Rolling train/test windows + robustness ratio | — |
| Live Chart | WebSocket candlestick streaming (1m–1W intervals) | — |
| News | NewsAPI + RSS feeds + StockTwits RSS | Free tier |
| Backend API | FastAPI + Pydantic v2 | Free |
| Frontend | React 18 + Vite + TailwindCSS + Recharts | Free |
| LLM | Groq API (Llama 3.3 70B) | Free (14,400 req/day) |

---

## 🚀 Dashboard — 8 Tabs

| Tab | Feature | Key Tech |
|-----|---------|---------|
| 🔍 **Analyze** | Full 7-agent AI analysis for any ticker | LangGraph, RAG, FinBERT |
| 💼 **Portfolio** | Real-time P&L tracker with live prices | yfinance, React state |
| 📊 **Compare** | Rank multiple tickers side-by-side | asyncio.gather parallel |
| ⚙️ **Optimize** | Grid-search best strategy parameters | Thread pool, caching |
| 🔔 **Alerts** | WebSocket real-time price alerts | ConnectionManager, asyncio |
| 📅 **Earnings** | Upcoming earnings calendar | yfinance calendar API |
| 🔬 **Validate** | Walk-forward validation (detect overfitting) | Robustness ratio |
| 📈 **Live** | Real-time candlestick chart (1m–1Y ranges) | WebSocket streaming |

---

## 🧠 Key Technical Concepts

### Phase 1: DSA Optimizations
- **Segment Tree** — O(log n) support/resistance detection vs O(n) brute force
- **Welford's Algorithm** — O(1) rolling std per tick vs O(w) recomputation
- **Sliding Window** — O(n) rolling metrics with cumsum trick
- **Online Indicators** — O(1) EMA/ZScore for live trading hot path (~0.5 μs)

### Phase 2: RAG Pipeline
- **SHA-256 Deduplication** — never re-embed the same document twice
- **Recursive Chunking** — paragraph→sentence→word hierarchy preserves context
- **MMR Reranking** — `score = λ×sim(query,d) - (1-λ)×max_sim(d,selected)` with λ=0.5
- **Metadata Filtering** — 200× faster search by filtering before vector comparison

### Phase 3: LangGraph Agents
- **TradingState TypedDict** — shared blackboard pattern (no direct agent communication)
- **Conditional Retry Edge** — RiskAgent → StrategyAgent loop (max 3 retries)
- **FinBERT Sentiment** — ProsusAI/finbert scores SEC filings + news → biases strategy selection
- **Fallback Explanation** — works without Groq key using rule-based logic

### Phase 4: MERN Dashboard
- **WebSocket ConnectionManager** — concurrent client broadcasting with dead-connection cleanup
- **Double-Checked Locking** — thread-safe singleton model loading (prevents race conditions)
- **Walk-Forward Validation** — rolling train/test windows, robustness ratio = OOS Sharpe / IS Sharpe
- **Live Candlestick Chart** — WebSocket streaming with custom SVG CandlestickBar shape

---

## 📁 Project Structure

```
quantmind/
├── backend/                    # Python AI/ML Engine
│   ├── config/
│   │   ├── settings.py         # Pydantic BaseSettings (env vars)
│   │   └── logging_config.py   # Async non-blocking logging
│   ├── engine/                 # Phase 1: DSA Trading Engine
│   │   ├── segment_tree.py     # O(log n) range queries (recursive)
│   │   ├── fast_segment_tree.py# O(log n) range queries (iterative, 2.5× faster)
│   │   ├── sliding_window.py   # O(n) rolling SMA/EMA/Sharpe
│   │   ├── metrics.py          # Sharpe, Sortino, VaR, CVaR, Drawdown
│   │   ├── online_indicators.py# O(1) per tick: EMA, ZScore, Sharpe (Welford's)
│   │   ├── backtester.py       # Historical simulation (yfinance, no look-ahead)
│   │   ├── walk_forward.py     # Walk-forward validation engine
│   │   ├── live_trader.py      # Real-time trading (Alpaca WebSocket)
│   │   └── strategies/
│   │       ├── base_strategy.py# Abstract: generate_signals + get_latest_signal
│   │       ├── momentum.py     # EMA crossover (batch + O(1) online)
│   │       ├── mean_reversion.py# Bollinger Bands (batch + O(1) online)
│   │       ├── rsi_strategy.py # RSI overbought/oversold
│   │       └── macd_strategy.py# Triple EMA MACD
│   ├── rag/                    # Phase 2: RAG Pipeline
│   │   ├── sources/
│   │   │   ├── base_loader.py  # Document + SHA-256 dedup + BaseLoader
│   │   │   ├── sec_loader.py   # SEC EDGAR (free, async, rate-limited)
│   │   │   ├── news_loader.py  # NewsAPI + 5 RSS feeds (concurrent)
│   │   │   ├── reddit_loader.py# StockTwits RSS + optional Reddit PRAW
│   │   │   └── pdf_loader.py   # Local PDFs (page-level chunks)
│   │   ├── chunker.py          # RecursiveChunker (overlap, semantic splits)
│   │   ├── embeddings.py       # sentence-transformers (local, free) + thread-safe singleton
│   │   ├── vector_store.py     # ChromaDB (persistent, metadata filter)
│   │   ├── retriever.py        # Semantic search + MMR reranking
│   │   └── ingestion.py        # Async pipeline orchestrator
│   ├── graph/                  # Phase 3: LangGraph State Machine
│   │   ├── state.py            # TradingState TypedDict (shared state, 7 agents)
│   │   └── workflow.py         # 7-node graph + conditional retry edge
│   ├── agents/                 # Phase 3: LangGraph Agent Nodes
│   │   ├── research_agent.py   # yfinance market data
│   │   ├── rag_agent.py        # Phase 2 RAG integration
│   │   ├── sentiment_agent.py  # FinBERT (ProsusAI/finbert) sentiment scoring
│   │   ├── strategy_agent.py   # 4-strategy selection (sentiment-biased)
│   │   ├── backtest_agent.py   # Phase 1 Backtester integration
│   │   ├── risk_agent.py       # Risk limits + retry logic (max 3)
│   │   └── explainer_agent.py  # Groq LLM + fallback explanation
│   └── api/                    # FastAPI REST + WebSocket API
│       ├── schemas.py          # Pydantic request/response models
│       ├── main.py             # FastAPI app + CORS + lifespan
│       └── routes/
│           ├── analysis.py     # POST /analyze
│           ├── compare.py      # POST /compare
│           ├── optimize.py     # POST /optimize
│           ├── portfolio.py    # GET/POST /portfolio
│           ├── alerts.py       # WebSocket /alerts/ws
│           ├── earnings.py     # GET /earnings
│           ├── ticker.py       # GET /ticker/search
│           ├── walk_forward.py # POST /walk-forward
│           └── live_chart.py   # WebSocket /live-chart/ws/{ticker}
│
└── client/                     # React Dashboard
    └── src/
        ├── App.jsx             # 8-tab navigation
        ├── hooks/useAnalysis.js# Custom hook (loading/error/result)
        ├── services/api.js     # Axios client → FastAPI
        └── components/
            ├── TickerSearch.jsx      # Search form + quick suggestions
            ├── TickerAutocomplete.jsx# Yahoo Finance autocomplete
            ├── SignalBadge.jsx       # BUY/SELL/HOLD colored badge
            ├── BacktestChart.jsx     # Equity curve (Recharts AreaChart)
            ├── MetricsTable.jsx      # Sharpe/VaR/Drawdown/WinRate grid
            ├── RAGExplainer.jsx      # AI explanation + citations + FinBERT
            ├── PortfolioTracker.jsx  # Real-time P&L tracker
            ├── CompareStocks.jsx     # Multi-ticker comparison
            ├── StrategyOptimizer.jsx # Grid-search optimizer
            ├── PriceAlerts.jsx       # WebSocket price alerts
            ├── EarningsCalendar.jsx  # Earnings calendar
            ├── WalkForwardAnalysis.jsx# Walk-forward validation UI
            └── LivePriceChart.jsx    # Real-time candlestick chart
```

---

## ⚡ Quick Start — Run the Full System

### Prerequisites
- Python 3.9+
- Node.js 18+
- No API keys needed for basic functionality

### 1. Python Backend Setup

```bash
cd quantmind/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run all tests first
python3 -m pytest tests/unit/ -v
# Expected: 453 passed in ~1s

# Start FastAPI server
uvicorn api.main:app --reload --port 8000 --reload-exclude ".venv"
# → http://localhost:8000/docs (Swagger UI)
```

### 2. React Dashboard Setup

```bash
cd quantmind/client
npm install
npm run dev
# → http://localhost:5173 (open in browser)
```

### 3. Use the Dashboard

1. Open **http://localhost:5173**
2. **🔍 Analyze tab**: Enter ticker `AAPL`, click **Run Analysis** → 7-agent pipeline runs
3. **📈 Live tab**: Select ticker + time range + interval → real-time candlestick chart
4. **🔬 Validate tab**: Enter ticker + strategy → walk-forward robustness test
5. **🔔 Alerts tab**: Set price alerts → WebSocket pushes notifications instantly

---

## 🔑 Optional API Keys (All Free)

| Service | URL | Limit | Used For |
|---------|-----|-------|---------|
| Groq | [console.groq.com](https://console.groq.com) | 14,400 req/day | AI explanations (Llama 3.3 70B) |
| NewsAPI | [newsapi.org](https://newsapi.org) | 100 req/day | Financial news |
| Alpaca | [alpaca.markets](https://alpaca.markets) | Unlimited paper | Live trading |

Copy `backend/.env.example` → `backend/.env` and fill in your keys.

---

## 📊 Performance

| Component | Metric | Value |
|-----------|--------|-------|
| Test Suite | 453 tests | ~1s |
| Segment Tree Query | O(log n) | ~6 μs (n=2520) |
| Online EMA Update | O(1) | ~0.5 μs |
| Signal Generation | O(1) | ~1 μs |
| Batch Embedding | 64 chunks | ~50ms (CPU) |
| ChromaDB Search | Filtered | ~10ms |
| Full Analysis | End-to-end | ~10-15s |
| Walk-Forward | JPM Momentum | Robustness 0.39 (OVERFITTED) |
| Live Chart | WebSocket history | 100 candles in <1s |

---

## 🧪 Walk-Forward Validation — Example Results

```
JPM Momentum Strategy (2022-2024):
  In-sample Sharpe:      1.84  (looks great in backtest)
  Out-of-sample Sharpe:  0.71  (much worse on unseen data)
  Robustness Ratio:      0.39  → OVERFITTED ⚠️

AAPL MACD Strategy (2022-2024):
  In-sample Sharpe:      1.12
  Out-of-sample Sharpe:  0.89
  Robustness Ratio:      0.79  → ROBUST ✅
```

---

## 🤖 FinBERT Sentiment — Example Results

```
AAPL:  sentiment_score=+0.254  label=BULLISH   confidence=0.81
MSFT:  sentiment_score=-0.917  label=BEARISH   confidence=0.94
RELIANCE.NS: sentiment_score=-0.071 label=NEUTRAL confidence=0.67
```

Sentiment directly influences strategy selection:
- BULLISH (+0.2) → lowers threshold for MACD (trend-following)
- BEARISH (-0.2) → pushes toward MeanReversion

---

## 🗺️ Roadmap

- [x] Phase 1: DSA Backtesting Engine (288 tests)
- [x] Phase 2: RAG Pipeline (116 tests)
- [x] Phase 3: LangGraph Multi-Agent Orchestration + FastAPI (49 tests)
- [x] Phase 4: MERN Dashboard (React + 8 tabs)
- [x] Phase 4+: FinBERT Sentiment Agent (7th LangGraph node)
- [x] Phase 4+: Walk-Forward Validation (robustness ratio)
- [x] Phase 4+: RSI + MACD strategies (4 total)
- [x] Phase 4+: Live Candlestick Chart (WebSocket, 6 time ranges, 7 intervals)
- [ ] Phase 5: CI/CD + Deployment (GitHub Actions + Render + Vercel)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ by [Mohan Pandey](https://github.com/MohanpandeyA)*
