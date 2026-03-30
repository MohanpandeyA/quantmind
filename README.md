# QuantMind — AI-Powered Algorithmic Trading Strategy Advisor

> **LangGraph + RAG + MERN + DSA** | **453 Tests Passing ✅** | **All 4 Phases Complete** | 100% Free to Run

A production-grade fintech system that combines **quantitative trading strategies**, **AI-powered document retrieval**, and **explainable recommendations** — built with professional code practices across a full-stack architecture.

---

## 🎯 What Problem It Solves

Most retail traders and small hedge funds face:
1. **Strategy Overload** — thousands of strategies exist, no intelligent selection
2. **Black-box AI** — existing tools give signals without explanation
3. **Knowledge Fragmentation** — financial knowledge scattered across SEC filings, news, earnings
4. **Backtesting Blindness** — traders backtest without understanding *why* a strategy worked

**QuantMind solves all four** with a 5-agent LangGraph pipeline that backtests strategies, retrieves real financial documents, and generates cited AI explanations.

---

## 🏗️ System Architecture

```
User Query: "Should I buy AAPL?"
         ↓
React Dashboard (port 5173)
         ↓ POST /api/analysis
Express Server (port 5000) → saves to MongoDB Atlas
         ↓ calls FastAPI
FastAPI (port 8000)
         ↓ runs LangGraph workflow
ResearchAgent  → yfinance: fetch AAPL price data (free, no key)
         ↓
RAGAgent       → SEC 10-K + News → ChromaDB → MMR retrieval
         ↓
StrategyAgent  → Select: Momentum (golden cross) or Mean Reversion
         ↓
BacktestAgent  → Run on 3 years of real data → Sharpe=1.4, MDD=12%
         ↓
RiskAgent      → VaR=2.1%, CVaR=3.4% → Risk approved (or retry)
         ↓
ExplainerAgent → "BUY signal. iPhone revenue grew 8% (Apple 10-K 2024).
                  Services hit record $24B. Momentum confirmed."
         ↓
React Dashboard shows:
  ├── 📈 BUY/SELL/HOLD signal badge
  ├── 📊 Equity curve chart (Recharts)
  ├── 📋 Metrics table (Sharpe, VaR, Drawdown, Win Rate)
  └── 🤖 AI explanation with source citations
```

---

## 📦 Tech Stack

| Layer | Technology | Cost |
|-------|-----------|------|
| AI Orchestration | LangGraph (5-agent state machine) | Free |
| RAG Pipeline | ChromaDB + sentence-transformers | Free (local) |
| Market Data | yfinance + SEC EDGAR API | Free (no key) |
| DSA Engine | Segment Tree O(log n) + Welford's O(1) | — |
| Live Trading | Alpaca WebSocket + async Python | Free (paper) |
| News | NewsAPI + RSS feeds | Free tier |
| Backend API | FastAPI + Pydantic | Free |
| Node.js Server | Express + Mongoose + Helmet | Free |
| Database | MongoDB Atlas | Free (512MB) |
| Frontend | React 18 + Vite + TailwindCSS + Recharts | Free |
| LLM | Groq API (Llama 3.1 70B) | Free (14,400 req/day) |

---

## 🚀 Project Status — All 4 Phases Complete

| Phase | Description | Status | Tests |
|-------|-------------|--------|-------|
| **Phase 1** | DSA Backtesting Engine | ✅ Complete | 288 |
| **Phase 2** | RAG Pipeline | ✅ Complete | 116 |
| **Phase 3** | LangGraph Agents + FastAPI | ✅ Complete | 49 |
| **Phase 4** | MERN Dashboard | ✅ Complete | — |

**Total: 453/453 tests passing in 1.12 seconds**

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
│   │   ├── live_trader.py      # Real-time trading (Alpaca WebSocket)
│   │   └── strategies/
│   │       ├── base_strategy.py# Abstract: generate_signals + get_latest_signal
│   │       ├── momentum.py     # MA crossover (batch + O(1) online)
│   │       └── mean_reversion.py# Bollinger Bands (batch + O(1) online)
│   ├── rag/                    # Phase 2: RAG Pipeline
│   │   ├── sources/
│   │   │   ├── base_loader.py  # Document + SHA-256 dedup + BaseLoader
│   │   │   ├── sec_loader.py   # SEC EDGAR (free, async, rate-limited)
│   │   │   ├── news_loader.py  # NewsAPI + 5 RSS feeds (concurrent)
│   │   │   └── pdf_loader.py   # Local PDFs (page-level chunks)
│   │   ├── chunker.py          # RecursiveChunker (overlap, semantic splits)
│   │   ├── embeddings.py       # sentence-transformers (local, free)
│   │   ├── vector_store.py     # ChromaDB (persistent, metadata filter)
│   │   ├── retriever.py        # Semantic search + MMR reranking
│   │   └── ingestion.py        # Async pipeline orchestrator
│   ├── graph/                  # Phase 3: LangGraph State Machine
│   │   ├── state.py            # TradingState TypedDict (shared state)
│   │   └── workflow.py         # 5-node graph + conditional retry edge
│   ├── agents/                 # Phase 3: LangGraph Agent Nodes
│   │   ├── research_agent.py   # yfinance market data
│   │   ├── rag_agent.py        # Phase 2 RAG integration
│   │   ├── strategy_agent.py   # Momentum vs MeanReversion selection
│   │   ├── backtest_agent.py   # Phase 1 Backtester integration
│   │   ├── risk_agent.py       # Risk limits + retry logic (max 3)
│   │   └── explainer_agent.py  # Groq LLM + fallback explanation
│   ├── api/                    # Phase 3: FastAPI REST API
│   │   ├── schemas.py          # Pydantic request/response models
│   │   ├── routes/analysis.py  # POST /analyze endpoint
│   │   └── main.py             # FastAPI app + CORS + lifespan
│   └── tests/unit/             # 453 tests, all passing
│
├── server/                     # Phase 4: Node.js Express Server
│   └── src/
│       ├── app.js              # Express + Helmet + CORS + rate limiting
│       ├── config/
│       │   ├── db.js           # MongoDB Atlas connection
│       │   └── logger.js       # Winston structured logging
│       ├── models/Analysis.js  # Mongoose schema for analysis history
│       ├── middleware/
│       │   └── errorHandler.js # Global error handler
│       ├── services/
│       │   └── pythonBridge.js # Axios proxy to FastAPI
│       └── routes/analysis.js  # POST/GET/DELETE /api/analysis
│
└── client/                     # Phase 4: React Dashboard
    └── src/
        ├── App.jsx             # Main layout (3-column responsive grid)
        ├── hooks/useAnalysis.js# Custom hook (loading/error/result)
        ├── services/api.js     # Axios client → Express server
        └── components/
            ├── TickerSearch.jsx # Search form + quick suggestions
            ├── SignalBadge.jsx  # BUY/SELL/HOLD colored badge
            ├── BacktestChart.jsx# Equity curve (Recharts AreaChart)
            ├── MetricsTable.jsx # Sharpe/VaR/Drawdown/WinRate grid
            └── RAGExplainer.jsx # AI explanation + citations
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
pip install fastapi uvicorn[standard]

# Run all tests first
python3 -m pytest tests/unit/ -v
# Expected: 453 passed in ~1s

# Start FastAPI server
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs (Swagger UI)
```

### 2. Node.js Server Setup

```bash
cd quantmind/server
npm install
npm run dev
# → http://localhost:5000/health
```

### 3. React Dashboard Setup

```bash
cd quantmind/client
npm install
npm run dev
# → http://localhost:5173 (open in browser)
```

### 4. Use the Dashboard

1. Open **http://localhost:5173**
2. Enter ticker: `AAPL`
3. Enter query: `Should I buy Apple stock given current market conditions?`
4. Click **Run Analysis**
5. Wait ~10-15 seconds for the full LangGraph pipeline
6. See: BUY/SELL/HOLD signal + equity curve + metrics + AI explanation

---

## 🔑 Optional API Keys (All Free)

| Service | URL | Limit | Used For |
|---------|-----|-------|---------|
| Groq | [console.groq.com](https://console.groq.com) | 14,400 req/day | AI explanations (Llama 3.1 70B) |
| NewsAPI | [newsapi.org](https://newsapi.org) | 100 req/day | Financial news |
| Alpaca | [alpaca.markets](https://alpaca.markets) | Unlimited paper | Live trading |
| MongoDB Atlas | [mongodb.com/atlas](https://mongodb.com/atlas) | 512MB free | Analysis history |

Copy `backend/.env.example` → `backend/.env` and fill in your keys.

---

## 🧠 Key Technical Concepts

### Phase 1: DSA Optimizations
- **Segment Tree** — O(log n) support/resistance detection vs O(n) brute force
- **Welford's Algorithm** — O(1) rolling std per tick vs O(w) recomputation
- **Sliding Window** — O(n) rolling metrics with cumsum trick
- **Online Indicators** — O(1) EMA/ZScore for live trading hot path (~1μs)

### Phase 2: RAG Pipeline
- **SHA-256 Deduplication** — never re-embed the same document twice
- **Recursive Chunking** — paragraph→sentence→word hierarchy preserves context
- **MMR Reranking** — diverse results, not just top-K similar chunks
- **Metadata Filtering** — 200× faster search by filtering before vector comparison

### Phase 3: LangGraph Agents
- **TradingState TypedDict** — shared blackboard pattern (no direct agent communication)
- **Conditional Retry Edge** — RiskAgent → StrategyAgent loop (max 3 retries)
- **Singleton Retriever** — sentence-transformers loaded once, reused across requests
- **Fallback Explanation** — works without Groq key using rule-based logic

### Phase 4: MERN Dashboard
- **BFF Pattern** — Express proxies FastAPI (handles CORS, auth, rate limiting)
- **Custom Hook** — `useAnalysis` separates API logic from UI components
- **Recharts** — declarative React charting built on D3.js
- **Graceful Degradation** — works without MongoDB (analysis history disabled)

---

## 📊 Performance

| Component | Metric | Value |
|-----------|--------|-------|
| Test Suite | 453 tests | 1.12s |
| Segment Tree Query | O(log n) | ~6 μs (n=2520) |
| Online EMA Update | O(1) | ~0.5 μs |
| Signal Generation | O(1) | ~1 μs |
| Batch Embedding | 64 chunks | ~50ms (CPU) |
| ChromaDB Search | Filtered | ~10ms |
| Full Analysis | End-to-end | ~10-15s |

---

## 🗺️ Roadmap

- [x] Phase 1: DSA Backtesting Engine (288 tests)
- [x] Phase 2: RAG Pipeline (116 tests)
- [x] Phase 3: LangGraph Multi-Agent Orchestration + FastAPI (49 tests)
- [x] Phase 4: MERN Dashboard (React + Express + MongoDB)
- [ ] Phase 5: CI/CD + Deployment (GitHub Actions + Render + Vercel)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ by [Mohan Pandey](https://github.com/MohanpandeyA)*
