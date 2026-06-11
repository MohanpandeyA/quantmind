# QuantMind — Setup Guide

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.9+ | [python.org](https://python.org) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) |
| Git | any | [git-scm.com](https://git-scm.com) |
| Docker (optional) | 24+ | [docker.com](https://docker.com) |

---

## Option A: Local Dev (Recommended for Development)

### 1. Clone the repo

```bash
git clone https://github.com/MohanpandeyA/quantmind.git
cd quantmind
```

### 2. Configure environment variables

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` and fill in your keys:

```bash
# Required for AI explanations (free at console.groq.com)
GROQ_API_KEY=gsk_your_key_here

# Optional — for live news in RAG pipeline (free at newsapi.org)
NEWS_API_KEY=your_newsapi_key_here

# Optional — for live trading tab (free paper trading at alpaca.markets)
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> **Note:** The app works without any API keys. Without `GROQ_API_KEY`, explanations use rule-based fallback logic. Without `NEWS_API_KEY`, RAG uses SEC EDGAR + RSS feeds only.

### 3. Python backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Verify all 453 tests pass
python -m pytest tests/unit/ -v
# Expected: 453 passed in ~2s

# Start FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 --reload-exclude ".venv"
```

FastAPI is ready when you see:
```
INFO:     Application startup complete.
```

- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 4. React frontend setup

Open a new terminal:

```bash
cd client
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Option B: Docker (One Command)

### Prerequisites
- Docker Desktop installed and running

### Setup

```bash
# Copy env file (Docker reads it automatically)
cp backend/.env.example backend/.env
# Edit backend/.env with your API keys

# Start everything (backend + frontend + MongoDB)
docker compose up

# Or run in background
docker compose up -d
```

Services start in order:
1. MongoDB (10s)
2. FastAPI backend (60-90s — loads FinBERT + sentence-transformers)
3. React frontend (after backend is healthy)

Access:
- React Dashboard: http://localhost:5173
- FastAPI Swagger: http://localhost:8000/docs

### Stop

```bash
docker compose down          # Stop containers
docker compose down -v       # Stop + delete volumes (clears ChromaDB data)
```

---

## Optional API Keys

All keys are free. The app works without them (with reduced functionality).

| Key | URL | Free Limit | Used For |
|-----|-----|-----------|---------|
| Groq | [console.groq.com](https://console.groq.com) | 14,400 req/day | AI explanations (Llama 3.3 70B) |
| NewsAPI | [newsapi.org](https://newsapi.org) | 100 req/day | Financial news in RAG |
| Alpaca | [alpaca.markets](https://alpaca.markets) | Unlimited paper | Live trading tab |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'api'`
You must run uvicorn from inside the `backend/` directory with the venv activated:
```bash
cd backend && source .venv/bin/activate && uvicorn api.main:app --reload
```

### FinBERT takes 30-40 seconds to load on first start
This is normal — the model is ~440MB. It's pre-loaded at startup so subsequent requests are fast.

### `SSL: CERTIFICATE_VERIFY_FAILED` on macOS
The [`engine/ssl_fix.py`](../backend/engine/ssl_fix.py) module handles this automatically. It's applied at import time in `api/main.py`.

### ChromaDB `collection already exists` error
Delete the data directory and restart:
```bash
rm -rf backend/data/chroma
```

### Port 8000 already in use
```bash
lsof -ti:8000 | xargs kill -9
```
