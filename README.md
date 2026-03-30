# QuantMind — AI-Powered Algorithmic Trading Strategy Advisor

> **LangGraph + RAG + MERN + DSA** | 404 Tests Passing ✅ | 100% Free to Run

A real-world fintech system that combines **quantitative trading strategies**, **AI-powered document retrieval**, and **explainable recommendations** — built with production-grade code practices.

---

## 🎯 What Problem It Solves

Most retail traders and small hedge funds face:
1. **Strategy Overload** — thousands of strategies exist, no intelligent selection
2. **Black-box AI** — existing tools give signals without explanation
3. **Knowledge Fragmentation** — financial knowledge scattered across SEC filings, news, earnings
4. **Backtesting Blindness** — traders backtest without understanding *why* a strategy worked

**QuantMind solves all four** by combining a DSA-optimized backtesting engine with a RAG pipeline that retrieves and cites real financial documents.

---

## 🏗️ Architecture

```
User Query: "Should I buy AAPL?"
         ↓
ResearchAgent  → yfinance: fetch AAPL price data
         ↓
RAGAgent       → SEC 10-K + News → ChromaDB → MMR retrieval
         ↓
StrategyAgent  → Select: Momentum (golden cross) or Mean Reversion
         ↓
BacktestAgent  → Run on 4 years of real data → Sharpe=1.4, MDD=12%
         ↓
RiskAgent      → VaR=2.1%, CVaR=3.4% → Risk approved
         ↓
ExplainerAgent → "BUY signal. iPhone revenue grew 8% (Apple 10-K 2024).
                  Services hit record $24B. Momentum confirmed."
```

---

## 📦 Tech Stack

| Layer | Technology | Cost |
|-------|-----------|------|
| AI Orchestration | LangGraph (Phase 3) | Free |
| RAG Pipeline | ChromaDB + sentence-transformers | Free (local) |
| Market Data | yfinance + SEC EDGAR API | Free (no key) |
| DSA Engine | Segment Tree O(log n) + Welford's O(1) | — |
| Live Trading | Alpaca WebSocket + async Python | Free (paper) |
| News | NewsAPI + RSS feeds | Free tier |
| Backend | FastAPI (Phase 3) | Free |
| Frontend | React + MERN (Phase 4) | Free |
| LLM | Groq API (Llama 3.1 70B) | Free (14,400 req/day) |

---

## 🚀 Project Status

| Phase | Description | Status | Tests |
|-------|-------------|--------|-------|
| **Phase 1** | DSA Backtesting Engine | ✅ Complete | 288 |
| **Phase 2** | RAG Pipeline | ✅ Complete | 116 |
| **Phase 3** | LangGraph Agents | 🔄 In Progress | — |
| **Phase 4** | MERN Dashboard | ⏳ Planned | — |

**Total: 404/404 tests passing in 1.01 seconds**

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
    └── tests/unit/             # 404 tests, all passing
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
# Expected: 404 passed in ~1s
```

### Run a Backtest (No API Keys Needed)

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

### Ingest Financial Documents (No API Keys Needed)

```python
import asyncio
from rag.ingestion import IngestionPipeline

async def main():
    pipeline = IngestionPipeline.create_default()
    report = await pipeline.ingest_ticker("AAPL")
    print(f"Stored {report.chunks_stored} chunks in {report.duration_seconds:.1f}s")

asyncio.run(main())
```

---

## 🔑 Optional API Keys (All Free)

| Service | URL | Limit | Used For |
|---------|-----|-------|---------|
| Groq | [console.groq.com](https://console.groq.com) | 14,400 req/day | LLM (Phase 3) |
| NewsAPI | [newsapi.org](https://newsapi.org) | 100 req/day | Financial news |
| Alpaca | [alpaca.markets](https://alpaca.markets) | Unlimited paper | Live trading |
| MongoDB Atlas | [mongodb.com/atlas](https://mongodb.com/atlas) | 512MB free | Portfolio storage |

Copy `.env.example` to `.env` and fill in your keys.

---

## 🧠 Key Technical Concepts

### DSA Optimizations
- **Segment Tree** — O(log n) support/resistance detection vs O(n) brute force
- **Welford's Algorithm** — O(1) rolling std per tick vs O(w) recomputation
- **Sliding Window** — O(n) rolling metrics with cumsum trick

### RAG Pipeline
- **SHA-256 Deduplication** — never re-embed the same document twice
- **Recursive Chunking** — paragraph→sentence→word hierarchy preserves context
- **MMR Reranking** — diverse results, not just top-K similar chunks
- **Metadata Filtering** — 200× faster search by filtering before vector comparison

### Live Trading
- **O(1) Signal Generation** — OnlineEMA updates in ~1μs per tick
- **Async Logging** — QueueHandler never blocks the trading hot path
- **Circuit Breaker** — IncrementalMetrics halts trading on drawdown breach

---

## 📊 Performance

| Component | Metric | Value |
|-----------|--------|-------|
| Test Suite | 404 tests | 1.01s |
| Segment Tree Query | O(log n) | ~6 μs (n=2520) |
| Online EMA Update | O(1) | ~0.5 μs |
| Signal Generation | O(1) | ~1 μs |
| Batch Embedding | 64 chunks | ~50ms (CPU) |
| ChromaDB Search | Filtered | ~10ms |

---

## 🗺️ Roadmap

- [x] Phase 1: DSA Backtesting Engine
- [x] Phase 2: RAG Pipeline
- [ ] Phase 3: LangGraph Multi-Agent Orchestration
- [ ] Phase 4: MERN Dashboard (React + Express + MongoDB)
- [ ] Phase 5: CI/CD + Deployment (Render + Vercel)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ by [Mohan Pandey](https://github.com/MohanpandeyA)*
