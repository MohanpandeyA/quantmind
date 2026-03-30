# QuantMind — AI-Powered Algorithmic Trading Strategy Advisor

> **LangGraph + RAG + MERN + DSA** | **453 Tests Passing ✅** | 100% Free to Run

A real-world fintech system that combines **quantitative trading strategies**, **AI-powered document retrieval**, and **explainable recommendations** — built with production-grade code practices.

---

## 🎯 What Problem It Solves

Most retail traders and small hedge funds face:
1. **Strategy Overload** — thousands of strategies exist, no intelligent selection
2. **Black-box AI** — existing tools give signals without explanation
3. **Knowledge Fragmentation** — financial knowledge scattered across SEC filings, news, earnings
4. **Backtesting Blindness** — traders backtest without understanding *why* a strategy worked

**QuantMind solves all four** by combining a DSA-optimized backtesting engine with a RAG pipeline and a LangGraph multi-agent system that retrieves and cites real financial documents.

---

## 🏗️ System Architecture

```
User Query: "Should I buy AAPL?"
         ↓
ResearchAgent  → yfinance: fetch AAPL price data (free, no key)
         ↓
RAGAgent       → SEC 10-K + News → ChromaDB → MMR retrieval
         ↓
StrategyAgent  → Select: Momentum (golden cross) or Mean Reversion
         ↓
BacktestAgent  → Run on 4 years of real data → Sharpe=1.4, MDD=12%
         ↓
RiskAgent      → VaR=2.1%, CVaR=3.4% → Risk approved (or retry)
         ↓
ExplainerAgent → "BUY signal. iPhone revenue grew 8% (Apple 10-K 2024).
                  Services hit record $24B. Momentum confirmed."
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
| Frontend | React + MERN (Phase 4) | Free |
| LLM | Groq API (Llama 3.1 70B) | Free (14,400 req/day) |

---

## 🚀 Project Status

| Phase | Description | Status | Tests |
|-------|-------------|--------|-------|
| **Phase 1** | DSA Backtesting Engine | ✅ Complete | 288 |
| **Phase 2** | RAG Pipeline | ✅ Complete | 116 |
| **Phase 3** | LangGraph Agents + FastAPI | ✅ Complete | 49 |
| **Phase 4** | MERN Dashboard | ⏳ Planned | — |

**Total: 453/453 tests passing in 1.12 seconds**

---

## 📁 Project Structure

```
quantmind/
└── backend/                    # Python AI/ML Engine
    ├── config/
    │   ├── settings.py         # Pydantic BaseSettings (env vars)
    │   └── logging_config.py   # Async non-blocking logging
    │
    ├── engine/                 # Phase 1: DSA Trading Engine
    │   ├── segment_tree.py     # O(log n) range queries (recursive)
    │   ├── fast_segment_tree.py# O(log n) range queries (iterative, 2.5× faster)
    │   ├── sliding_window.py   # O(n) rolling SMA/EMA/Sharpe
    │   ├── metrics.py          # Sharpe, Sortino, VaR, CVaR, Drawdown
    │   ├── online_indicators.py# O(1) per tick: EMA, ZScore, Sharpe (Welford's)
    │   ├── backtester.py       # Historical simulation (yfinance, no look-ahead)
    │   ├── live_trader.py      # Real-time trading (Alpaca WebSocket)
    │   └── strategies/
    │       ├── base_strategy.py# Abstract: generate_signals + get_latest_signal
    │       ├── momentum.py     # MA crossover (batch + O(1) online)
    │       └── mean_reversion.py# Bollinger Bands (batch + O(1) online)
    │
    ├── rag/                    # Phase 2: RAG Pipeline
    │   ├── sources/
    │   │   ├── base_loader.py  # Document + SHA-256 dedup + BaseLoader
    │   │   ├── sec_loader.py   # SEC EDGAR (free, async, rate-limited)
    │   │   ├── news_loader.py  # NewsAPI + 5 RSS feeds (concurrent)
    │   │   └── pdf_loader.py   # Local PDFs (page-level chunks)
    │   ├── chunker.py          # RecursiveChunker (overlap, semantic splits)
    │   ├── embeddings.py       # sentence-transformers (local, free)
    │   ├── vector_store.py     # ChromaDB (persistent, metadata filter)
    │   ├── retriever.py        # Semantic search + MMR reranking
    │   └── ingestion.py        # Async pipeline orchestrator
    │
    ├── graph/                  # Phase 3: LangGraph State Machine
    │   ├── state.py            # TradingState TypedDict (shared state)
    │   └── workflow.py         # 5-node graph + conditional retry edge
    │
    ├── agents/                 # Phase 3: LangGraph Agent Nodes
    │   ├── research_agent.py   # yfinance market data
    │   ├── rag_agent.py        # Phase 2 RAG integration
    │   ├── strategy_agent.py   # Momentum vs MeanReversion selection
    │   ├── backtest_agent.py   # Phase 1 Backtester integration
    │   ├── risk_agent.py       # Risk limits + retry logic (max 3)
    │   └── explainer_agent.py  # Groq LLM + fallback explanation
    │
    ├── api/                    # Phase 3: FastAPI REST API
    │   ├── schemas.py          # Pydantic request/response models
    │   ├── routes/analysis.py  # POST /analyze endpoint
    │   └── main.py             # FastAPI app + CORS + lifespan
    │
    └── tests/unit/             # 453 tests, all passing
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.9+
- No API keys needed for basic functionality

### Setup

```bash
# Clone the repository
git clone https://github.com/MohanpandeyA/quantmind.git
cd quantmind/backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run all tests
python3 -m pytest tests/unit/ -v
# Expected: 453 passed in ~1s
```

### Run the FastAPI Server

```bash
cd quantmind/backend
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

Then open:
- **API Docs:** http://localhost:8000/docs (Swagger UI)
- **Health:** http://localhost:8000/health

### Test the Full Analysis Pipeline

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "query": "Should I buy Apple stock given current market conditions?",
    "start_date": "2022-01-01",
    "end_date": "2024-12-31"
  }'
```

**Response:**
```json
{
  "ticker": "AAPL",
  "signal": "BUY",
  "final_explanation": "SIGNAL: BUY\nCONFIDENCE: MEDIUM\n\nApple shows...",
  "backtest_results": {
    "sharpe_ratio": 1.42,
    "total_return": 0.28,
    "max_drawdown": 0.12
  },
  "processing_time_ms": 8340.5
}
```

### Run a Backtest Directly (No API Keys Needed)

```python
from engine.backtester import Backtester, BacktestConfig
from engine.strategies.momentum import MomentumStrategy
from engine.strategies.base_strategy import StrategyConfig

strategy = MomentumStrategy(StrategyConfig(
    params={"short_window": 20, "long_window": 50}
))

bt = Backtester(
    BacktestConfig("AAPL", "2020-01-01", "2024-12-31"),
    strategy
)
result, report = bt.run()
print(f"Sharpe: {report.sharpe_ratio:.2f}")
print(f"Total Return: {report.total_return:.1%}")
print(f"Max Drawdown: {report.max_drawdown:.1%}")
```

---

## 🔑 Optional API Keys (All Free)

| Service | URL | Limit | Used For |
|---------|-----|-------|---------|
| Groq | [console.groq.com](https://console.groq.com) | 14,400 req/day | LLM explanations |
| NewsAPI | [newsapi.org](https://newsapi.org) | 100 req/day | Financial news |
| Alpaca | [alpaca.markets](https://alpaca.markets) | Unlimited paper | Live trading |
| MongoDB Atlas | [mongodb.com/atlas](https://mongodb.com/atlas) | 512MB free | Portfolio storage (Phase 4) |

Copy `.env.example` to `.env` and fill in your keys.

---

## 🧠 Key Technical Concepts

### Phase 1: DSA Optimizations
- **Segment Tree** — O(log n) support/resistance detection vs O(n) brute force
- **Welford's Algorithm** — O(1) rolling std per tick vs O(w) recomputation
- **Sliding Window** — O(n) rolling metrics with cumsum trick
- **Online Indicators** — O(1) EMA/ZScore for live trading hot path

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
| Full Analysis | End-to-end | ~8-15s |

---

## 🗺️ Roadmap

- [x] Phase 1: DSA Backtesting Engine (288 tests)
- [x] Phase 2: RAG Pipeline (116 tests)
- [x] Phase 3: LangGraph Multi-Agent Orchestration + FastAPI (49 tests)
- [ ] Phase 4: MERN Dashboard (React + Express + MongoDB)
- [ ] Phase 5: CI/CD + Deployment (Render + Vercel)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ by [Mohan Pandey](https://github.com/MohanpandeyA)*
