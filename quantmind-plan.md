# QuantMind — AI-Powered Algorithmic Trading Strategy Advisor

## Project Overview

**Problem Solved:** Retail traders and small hedge funds lack an intelligent, explainable system that:
1. Selects the right trading strategy based on current market conditions
2. Explains *why* using real financial documents (SEC filings, earnings, news)
3. Backtests efficiently using DSA-optimized algorithms
4. Presents everything in a clean, real-time dashboard

**Core Tech:** LangGraph + RAG + MERN + DSA (Segment Trees, Sliding Window)

---

## Free Tier Stack (Start Here) → Paid Upgrade Path

| Component | Free Tier | Paid Upgrade |
|-----------|-----------|--------------|
| LLM | Groq API (Llama 3.1 70B — free) | OpenAI GPT-4o / Claude 3.5 |
| Embeddings | nomic-embed-text via Ollama (local) | OpenAI text-embedding-3-large |
| Vector DB | ChromaDB (local, persistent) | Pinecone / Weaviate cloud |
| Market Data | yfinance (Yahoo Finance, free) | Alpha Vantage Pro / Polygon.io |
| News/Docs | NewsAPI (free 100 req/day) + RSS | Bloomberg API / Refinitiv |
| SEC Filings | SEC EDGAR API (always free) | Same |
| MongoDB | MongoDB Atlas Free Tier (512MB) | Atlas M10+ |
| Deployment | Railway.app / Render.com free | AWS / GCP / Azure |
| Frontend | Vercel free tier | Same |
| LangGraph | Open source (free) | Same |

---

## Free API Keys Required

| Service | URL | Limit |
|---------|-----|-------|
| Groq | https://console.groq.com | 14,400 req/day free |
| NewsAPI | https://newsapi.org | 100 req/day free |
| MongoDB Atlas | https://mongodb.com/atlas | 512MB free forever |
| SEC EDGAR | No key needed | Unlimited |
| yfinance | No key needed (pip install) | Unlimited |

---

## Project Structure

```
quantmind/
├── backend/                          # Python - AI/ML Engine
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── research_agent.py         # Fetches market data
│   │   ├── rag_agent.py              # RAG retrieval agent
│   │   ├── strategy_agent.py         # Strategy selection
│   │   ├── backtest_agent.py         # Runs backtesting
│   │   ├── risk_agent.py             # Risk assessment
│   │   └── explainer_agent.py        # Generates explanation
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── workflow.py               # LangGraph state machine
│   │   └── state.py                  # TypedDict state definitions
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingestion.py              # Document loading + chunking
│   │   ├── embeddings.py             # Embedding model wrapper
│   │   ├── retriever.py              # ChromaDB retrieval
│   │   └── sources/
│   │       ├── sec_loader.py         # SEC EDGAR loader
│   │       ├── news_loader.py        # NewsAPI + RSS loader
│   │       └── pdf_loader.py         # PDF document loader
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── backtester.py             # Main backtesting engine
│   │   ├── segment_tree.py           # DSA: O(log n) range queries
│   │   ├── sliding_window.py         # DSA: Rolling metrics
│   │   ├── strategies/
│   │   │   ├── momentum.py
│   │   │   ├── mean_reversion.py
│   │   │   └── base_strategy.py      # Abstract base class
│   │   └── metrics.py                # Sharpe, Drawdown, VaR
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app
│   │   ├── routes/
│   │   │   ├── analysis.py           # POST /analyze
│   │   │   ├── backtest.py           # POST /backtest
│   │   │   └── health.py             # GET /health
│   │   └── schemas.py                # Pydantic models
│   ├── config/
│   │   ├── settings.py               # Pydantic BaseSettings
│   │   └── logging_config.py         # Structured logging
│   ├── tests/
│   │   ├── unit/
│   │   │   ├── test_segment_tree.py
│   │   │   ├── test_sliding_window.py
│   │   │   └── test_strategies.py
│   │   └── integration/
│   │       ├── test_rag_pipeline.py
│   │       └── test_langgraph_flow.py
│   ├── .env.example
│   ├── requirements.txt
│   └── pyproject.toml
│
├── server/                           # Node.js - MERN Backend
│   ├── src/
│   │   ├── controllers/
│   │   │   ├── analysisController.js
│   │   │   └── portfolioController.js
│   │   ├── models/
│   │   │   ├── Portfolio.js
│   │   │   ├── Analysis.js
│   │   │   └── Strategy.js
│   │   ├── routes/
│   │   │   ├── analysis.js
│   │   │   └── portfolio.js
│   │   ├── middleware/
│   │   │   ├── errorHandler.js
│   │   │   ├── rateLimiter.js
│   │   │   └── validator.js
│   │   ├── services/
│   │   │   └── pythonBridge.js
│   │   ├── config/
│   │   │   └── db.js
│   │   └── app.js
│   ├── tests/
│   │   └── api.test.js
│   ├── .env.example
│   └── package.json
│
├── client/                           # React Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard/
│   │   │   ├── StrategyCard/
│   │   │   ├── BacktestChart/
│   │   │   └── RAGExplainer/
│   │   ├── hooks/
│   │   │   └── useAnalysis.js
│   │   ├── services/
│   │   │   └── api.js
│   │   └── App.jsx
│   └── package.json
│
├── docs/
│   ├── architecture.md
│   ├── api-reference.md
│   └── setup.md
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
└── README.md
```

---

## LangGraph Agent Workflow

### State Definition (state.py)
```python
from typing import TypedDict, Optional, List
from dataclasses import dataclass

class TradingState(TypedDict):
    ticker: str
    market_data: Optional[dict]
    retrieved_docs: Optional[List[str]]
    selected_strategy: Optional[str]
    backtest_results: Optional[dict]
    risk_metrics: Optional[dict]
    risk_approved: bool
    retry_count: int
    final_explanation: Optional[str]
    sources: Optional[List[str]]
```

### Agent Flow
```
START
  → ResearchAgent    (fetch OHLCV data via yfinance)
  → RAGAgent         (retrieve relevant SEC/news docs from ChromaDB)
  → StrategyAgent    (select best strategy based on market + docs)
  → BacktestAgent    (run DSA-optimized backtest)
  → RiskAgent        (compute Sharpe, VaR, max drawdown)
    → if risk too high → back to StrategyAgent (max 3 retries)
    → if risk OK → ExplainerAgent
  → ExplainerAgent   (generate cited explanation)
END
```

---

## DSA Components

### Segment Tree (segment_tree.py)
- **Purpose:** O(log n) range max/min queries on price arrays
- **Use case:** Find support/resistance levels, highest high / lowest low in any date range
- **Operations:** `build()`, `query(l, r)`, `update(i, val)`

### Sliding Window (sliding_window.py)
- **Purpose:** O(n) rolling metrics computation
- **Use case:** Rolling Sharpe ratio, rolling volatility, moving averages
- **Operations:** `rolling_mean()`, `rolling_std()`, `rolling_sharpe()`

### Priority Queue / Heap
- **Purpose:** Real-time top-N strategy ranking
- **Use case:** Always surface the best-performing strategy

### Graph (Correlation Matrix)
- **Purpose:** Asset correlation for portfolio diversification
- **Use case:** Detect correlated clusters, avoid over-concentration

---

## RAG Pipeline

### Document Sources
1. **SEC EDGAR** — 10-K, 10-Q filings (free, no key)
2. **NewsAPI** — Financial news (100 req/day free)
3. **RSS Feeds** — Reuters, Bloomberg RSS (free)
4. **User-uploaded PDFs** — Research papers, analyst reports

### Pipeline Steps
1. Load documents via source-specific loaders
2. Chunk text (RecursiveCharacterTextSplitter, chunk_size=1000, overlap=200)
3. Embed chunks (nomic-embed-text via Ollama OR Groq)
4. Store in ChromaDB with metadata (source, date, ticker)
5. On query: embed query → semantic search → top-K chunks → LLM with context
6. Return answer + source citations

---

## Code Quality Standards

### Python
- **black** — auto-formatter (line length 88)
- **isort** — import sorting
- **mypy** — static type checking (strict mode)
- **pytest** — unit + integration tests (target 80%+ coverage)
- **pydantic** — all data validation, no raw dicts
- **Google-style docstrings** on every public function/class
- **.env never committed** — python-dotenv + .env.example template

### Node.js
- **eslint + prettier** — formatting
- **jest** — API tests
- **joi** — request validation
- **winston** — structured logging
- **express-rate-limit** — protect endpoints
- **helmet** — security headers

### Git Practices
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`
- GitHub Actions CI — runs tests on every PR
- Branch strategy: `main` → `develop` → `feature/xxx`
- Never commit secrets — use .env.example

---

## Implementation Phases

### Phase 1: DSA Backtesting Engine (Python)
- [ ] Implement `SegmentTree` class with build/query/update
- [ ] Implement `SlidingWindow` utilities (rolling mean, std, Sharpe)
- [ ] Implement `BaseStrategy` abstract class
- [ ] Implement `MomentumStrategy` and `MeanReversionStrategy`
- [ ] Implement `Backtester` class using yfinance data
- [ ] Implement `metrics.py` (Sharpe ratio, max drawdown, VaR)
- [ ] Write unit tests for all DSA components
- [ ] Write unit tests for strategies

### Phase 2: RAG Pipeline
- [ ] Set up ChromaDB persistent store
- [ ] Implement `sec_loader.py` using SEC EDGAR API
- [ ] Implement `news_loader.py` using NewsAPI + feedparser
- [ ] Implement `pdf_loader.py` using LangChain PDF loader
- [ ] Implement `ingestion.py` — chunking + embedding pipeline
- [ ] Implement `retriever.py` — semantic search wrapper
- [ ] Write integration tests for RAG pipeline
- [ ] Test with real SEC filings (AAPL, MSFT 10-K)

### Phase 3: LangGraph Agent Orchestration
- [ ] Define `TradingState` TypedDict in `state.py`
- [ ] Implement each agent as a LangGraph node function
- [ ] Build `workflow.py` — connect nodes with conditional edges
- [ ] Implement retry logic in RiskAgent → StrategyAgent loop
- [ ] Expose workflow via FastAPI (`POST /analyze`)
- [ ] Add Pydantic schemas for all request/response models
- [ ] Write integration tests for full agent flow

### Phase 4: MERN Layer
- [ ] Set up MongoDB Atlas free tier + Mongoose schemas
- [ ] Build Express API (analysis, portfolio routes)
- [ ] Implement `pythonBridge.js` to call FastAPI
- [ ] Build React dashboard with Recharts
- [ ] Build `RAGExplainer` component showing source citations
- [ ] Build `BacktestChart` component with equity curve
- [ ] Add rate limiting, error handling middleware

### Phase 5: Tests, CI/CD, Docs
- [ ] Set up GitHub Actions CI (pytest + jest on PR)
- [ ] Write docker-compose.yml for local dev
- [ ] Write README.md with setup instructions
- [ ] Write API reference docs
- [ ] Deploy backend to Render.com free tier
- [ ] Deploy frontend to Vercel free tier

---

## Environment Variables Template (.env.example)

```bash
# LLM
GROQ_API_KEY=your_groq_api_key_here

# News
NEWS_API_KEY=your_newsapi_key_here

# MongoDB
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/quantmind

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma

# FastAPI
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000

# Node.js
PORT=5000
PYTHON_API_URL=http://localhost:8000

# When you upgrade (leave empty for now)
OPENAI_API_KEY=
PINECONE_API_KEY=
POLYGON_API_KEY=
```

---

## Upgrade Path (When You Have Budget)

| Current (Free) | Upgrade To | Benefit |
|----------------|-----------|---------|
| Groq Llama 3.1 70B | GPT-4o / Claude 3.5 | Better reasoning |
| ChromaDB local | Pinecone cloud | Scalable, managed |
| yfinance | Polygon.io | Real-time tick data |
| Render.com free | AWS ECS / GCP Cloud Run | Production SLA |
| MongoDB Atlas free | Atlas M10 | More storage + ops |

The codebase is designed so each of these is a **single config change** — no refactoring needed.
