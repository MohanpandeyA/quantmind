# QuantMind — System Architecture

## High-Level Overview

```
User Browser
     │
     │  HTTP/WebSocket
     ▼
React Dashboard (Vite, port 5173)
     │
     │  /api/* → proxy → port 8000
     ▼
FastAPI (Python, port 8000)
     │
     ├── POST /analyze ──────────────────────────────────────────────────────┐
     │                                                                        │
     │                         LangGraph State Machine                        │
     │   ┌──────────────────────────────────────────────────────────────┐    │
     │   │                                                              │    │
     │   │  ResearchAgent → RAGAgent → SentimentAgent → StrategyAgent  │    │
     │   │       │              │            │               │          │    │
     │   │   yfinance       ChromaDB      FinBERT        4 strategies  │    │
     │   │                  (1784 docs)  (ProsusAI)    (Momentum/RSI/  │    │
     │   │                                              MACD/MeanRev)  │    │
     │   │                                                              │    │
     │   │  BacktestAgent → RiskAgent ──────────────────────────────── │    │
     │   │       │              │    ↑ retry (max 3)                   │    │
     │   │  SegmentTree     Sharpe/VaR/                                 │    │
     │   │  SlidingWindow   Drawdown                                    │    │
     │   │                                                              │    │
     │   │  ExplainerAgent → END                                        │    │
     │   │       │                                                      │    │
     │   │   Groq LLM                                                   │    │
     │   │   (Llama 3.3 70B)                                            │    │
     │   └──────────────────────────────────────────────────────────────┘    │
     │                                                                        │
     └────────────────────────────────────────────────────────────────────────┘
     │
     ├── WS /alerts/ws ──── ConnectionManager ──── yfinance price polling
     ├── WS /live-chart/ws/{ticker} ──── yfinance streaming
     ├── POST /compare ──── asyncio.gather (parallel analysis)
     ├── POST /optimize ─── ThreadPoolExecutor (grid search)
     └── POST /walk-forward ─ rolling train/test windows
```

---

## LangGraph 7-Agent State Machine

```
TradingState (shared blackboard — TypedDict)
┌─────────────────────────────────────────────────────────────────┐
│ ticker, query, start_date, end_date                             │
│ market_data, price_history                                      │
│ rag_context, retrieved_docs                                     │
│ sentiment_score, sentiment_label, sentiment_confidence          │
│ selected_strategy, strategy_params, strategy_rationale          │
│ backtest_results, equity_curve                                  │
│ risk_metrics, risk_approved, retry_count                        │
│ signal, final_explanation, final_citations                      │
│ processing_time_ms, error                                       │
└─────────────────────────────────────────────────────────────────┘

START
  │
  ▼
┌─────────────────┐
│  ResearchAgent  │  yfinance.download() → OHLCV bars
│                 │  Computes: price_change_pct, 52w high/low, volume
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    RAGAgent     │  ChromaDB semantic search (MMR reranking)
│                 │  Sources: SEC 10-K/10-Q, NewsAPI, RSS, PDFs
│                 │  SHA-256 dedup — never re-embeds same doc
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SentimentAgent  │  FinBERT (ProsusAI/finbert) on RAG context
│                 │  Output: score ∈ [-1, +1], label, confidence
│                 │  BULLISH → biases toward MACD
│                 │  BEARISH → biases toward MeanReversion
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ StrategyAgent   │  4-way decision tree:
│                 │  Trending + strong + bullish → MACD
│                 │  Trending + moderate         → Momentum
│                 │  Oscillating + high vol      → MeanReversion
│                 │  Oscillating + low vol       → RSI
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BacktestAgent  │  Runs selected strategy on 3 years of data
│                 │  Uses SegmentTree for O(log n) range queries
│                 │  Caches yfinance data per ticker per date range
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   RiskAgent     │  Checks: Sharpe ≥ 0.5, MDD ≤ 25%, WinRate ≥ 30%
│                 │  Computes: VaR 95%, CVaR 95%, risk score
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
  PASS      FAIL (retry_count < 3)
    │         │
    │         └──────────────────────────────────────────────────┐
    │                                                             │
    ▼                                                             │
┌─────────────────┐                                              │
│ ExplainerAgent  │  Groq Llama 3.3 70B generates cited answer  │
│                 │  Fallback: rule-based if no GROQ_API_KEY     │
│                 │  Output: signal (BUY/SELL/HOLD) + explanation│
└────────┬────────┘                                              │
         │                                                        │
        END ◄───────────────────────────────────────────────────┘
                                              (after 3 retries, approve with warning)
```

---

## RAG Pipeline

```
Document Sources
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  SEC EDGAR   │  │   NewsAPI    │  │  RSS Feeds   │  │  Local PDFs  │
│  10-K, 10-Q  │  │  100/day     │  │  Reuters,    │  │  Research    │
│  (free, no   │  │  (free tier) │  │  Bloomberg   │  │  papers      │
│   key needed)│  │              │  │  StockTwits  │  │              │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │                  │
       └─────────────────┴──────────────────┴──────────────────┘
                                    │
                                    ▼
                         ┌──────────────────┐
                         │  SHA-256 Dedup   │  Never re-embed same doc
                         │  base_loader.py  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ RecursiveChunker │  chunk_size=1000, overlap=200
                         │  chunker.py      │  paragraph→sentence→word
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ EmbeddingModel   │  sentence-transformers
                         │  embeddings.py   │  all-MiniLM-L6-v2 (384 dims)
                         │                  │  Thread-safe singleton
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    ChromaDB      │  Persistent local vector store
                         │  vector_store.py │  1,784 docs indexed
                         │                  │  Metadata filter (200× faster)
                         └────────┬─────────┘
                                  │
                         On query:│
                                  ▼
                         ┌──────────────────┐
                         │  MMR Retriever   │  score = λ×sim(q,d) - (1-λ)×max_sim(d,S)
                         │  retriever.py    │  λ=0.5, top-5 diverse results
                         └──────────────────┘
```

---

## DSA Components

```
Segment Tree (segment_tree.py / fast_segment_tree.py)
─────────────────────────────────────────────────────
Purpose: O(log n) range max/min queries on price arrays
Use case: Support/resistance levels, highest high / lowest low in any date range

Array:  [150, 155, 148, 162, 158, 170, 165]
Tree:   [170, 162, 170, 155, 162, 170, 165]
         root  left  right ...

query(2, 5) → max in range [148, 162, 158, 170] = 170  ← O(log n) vs O(n) brute force

Two trees built per backtest: max_tree (resistance) + min_tree (support)


Sliding Window (sliding_window.py)
───────────────────────────────────
Purpose: O(n) rolling metrics using cumsum trick
Operations: rolling_mean(), rolling_std(), rolling_sharpe()

rolling_mean(window=20):
  cumsum[i] = sum(prices[0..i])
  mean[i]   = (cumsum[i] - cumsum[i-20]) / 20  ← O(1) per step


Online Indicators (online_indicators.py)
─────────────────────────────────────────
Purpose: O(1) per tick for live trading hot path (~0.5 μs)
Uses Welford's algorithm for numerically stable online variance

OnlineEMA.update(price):   α = 2/(period+1); ema = α*price + (1-α)*ema_prev
OnlineZScore.update(x):    Welford's: mean, M2 updated in O(1)
OnlineSharpe.update(ret):  Welford's variance on returns stream


Metrics (metrics.py)
─────────────────────
Sharpe  = (mean_return - risk_free) / std_return × √252
Sortino = (mean_return - risk_free) / downside_std × √252
VaR 95% = percentile(returns, 5%)
CVaR 95% = mean(returns[returns < VaR])
Max DD  = max(peak - trough) / peak over all windows
```

---

## WebSocket Architecture

```
Price Alerts (/alerts/ws)
──────────────────────────
ConnectionManager (singleton)
├── active_connections: Set[WebSocket]
├── alerts: List[PriceAlert]
└── _monitor_task: asyncio.Task (polls yfinance every 30s)

Client connects → accept() → send welcome JSON → add to active_connections
Client disconnects → discard from active_connections
Price monitor fires → broadcast() to all active_connections
                    → dead connections cleaned up silently


Live Chart (/live-chart/ws/{ticker})
──────────────────────────────────────
Per-connection handler (no shared state):
1. Accept WebSocket
2. Fetch 100 historical candles → send {"type": "history", "candles": [...]}
3. Loop every 5s:
   a. Fetch latest candle from yfinance
   b. Send {"type": "update", "candle": {...}}
4. On disconnect → exit loop cleanly
```

---

## Deployment Architecture

```
GitHub (source of truth)
        │
        │  git push
        ▼
GitHub Actions CI
├── pytest tests/unit/ (453 tests)
├── black --check + isort --check
├── npm run build (Vite)
└── docker build (validate Dockerfile)
        │
        │  on main branch
        ├──────────────────────────────────────────────────────┐
        ▼                                                       ▼
Render.com (backend)                                    Vercel (frontend)
├── Python 3.11                                         ├── Vite build
├── uvicorn api.main:app                                ├── Static /dist
├── Port: $PORT (10000)                                 ├── CDN edge network
├── ChromaDB: /tmp/chroma                               └── VITE_API_URL →
└── Env vars: Render dashboard                              Render backend URL
        │                                                       │
        └───────────────────────────────────────────────────────┘
                              │
                    User browser hits
                    quantmind.vercel.app
                    /api/* → onrender.com
```

---

## File Structure

```
quantmind/
├── .github/workflows/ci.yml    ← GitHub Actions (pytest + lint + docker build)
├── docker-compose.yml          ← One-command local dev
├── render.yaml                 ← Render.com deployment config
├── vercel.json                 ← Vercel deployment config
├── docs/
│   ├── setup.md                ← Local setup guide
│   ├── api-reference.md        ← All 11 endpoints documented
│   └── architecture.md         ← This file
├── backend/
│   ├── Dockerfile              ← Multi-stage Python image
│   ├── requirements.txt
│   ├── config/                 ← Pydantic settings + logging
│   ├── engine/                 ← Phase 1: DSA (SegmentTree, SlidingWindow, etc.)
│   ├── rag/                    ← Phase 2: ChromaDB + sentence-transformers
│   ├── agents/                 ← Phase 3: 7 LangGraph agent nodes
│   ├── graph/                  ← Phase 3: workflow.py + state.py
│   ├── api/                    ← FastAPI routes + schemas
│   └── tests/unit/             ← 453 unit tests
└── client/
    ├── Dockerfile              ← Multi-stage Node → nginx image
    ├── nginx.conf              ← SPA routing + gzip + security headers
    └── src/
        ├── App.jsx             ← 8-tab dashboard
        ├── components/         ← 13 React components
        ├── hooks/              ← useAnalysis custom hook
        └── services/api.js     ← Axios client
```
