# QuantMind API Reference

Base URL (local): `http://localhost:8000`  
Base URL (production): `https://quantmind-backend.onrender.com`

Interactive docs: `{base_url}/docs` (Swagger UI)

---

## Health

### `GET /health`

Check if the API is running.

**Response `200`**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "phase": "Phase 3 — LangGraph Agents"
}
```

---

## Analysis

### `POST /analyze`

Run the full 7-agent LangGraph pipeline for a single ticker.

**Request body**
```json
{
  "ticker": "AAPL",
  "query": "Should I buy Apple stock?",
  "start_date": "2022-01-01",
  "end_date": "2024-12-31"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `ticker` | string | ✅ | — | Stock ticker (e.g. `AAPL`, `RELIANCE.NS`) |
| `query` | string | ✅ | — | Natural language question |
| `start_date` | string | ❌ | `2022-01-01` | Backtest start (YYYY-MM-DD) |
| `end_date` | string | ❌ | `2024-12-31` | Backtest end (YYYY-MM-DD) |

**Response `200`**
```json
{
  "ticker": "AAPL",
  "signal": "BUY",
  "selected_strategy": "mean_reversion",
  "strategy_rationale": "Market is oscillating...",
  "sentiment_score": 0.254,
  "sentiment_label": "BULLISH",
  "sentiment_confidence": 0.81,
  "sentiment_details": [{"text": "...", "score": 0.81, "label": "positive"}],
  "backtest_results": {
    "sharpe_ratio": 0.71,
    "total_return": 0.436,
    "max_drawdown": 0.165,
    "win_rate": 0.72,
    "n_trades": 25,
    "start_date": "2022-01-01",
    "end_date": "2024-12-31"
  },
  "risk_metrics": {
    "var_95": 0.019,
    "cvar_95": 0.028,
    "risk_level": "MEDIUM",
    "risk_score": 5.3,
    "risk_approved": true
  },
  "equity_curve": [
    {"date": "2022-01-03", "value": 100000.0},
    {"date": "2022-01-04", "value": 100234.5}
  ],
  "final_explanation": "SIGNAL: BUY\nCONFIDENCE: MEDIUM\n\nApple shows...",
  "final_citations": ["Apple 10-K 2024", "Reuters 2024-01-15"],
  "market_data": {
    "current_price": 291.58,
    "price_change_pct": 0.35,
    "volume": 52341200,
    "week_52_high": 320.0,
    "week_52_low": 164.08
  },
  "processing_time_ms": 6712.0
}
```

**Typical latency:** 6–15 seconds (depends on Groq API response time)

---

### `POST /compare`

Rank multiple tickers side-by-side using parallel analysis.

**Request body**
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "query": "Which stock should I buy?",
  "start_date": "2022-01-01",
  "end_date": "2024-12-31"
}
```

**Response `200`**
```json
{
  "rankings": [
    {
      "rank": 1,
      "ticker": "MSFT",
      "signal": "HOLD",
      "composite_score": 0.40,
      "sharpe_ratio": 0.71,
      "total_return": 0.436,
      "max_drawdown": 0.165,
      "win_rate": 0.72,
      "strategy": "mean_reversion",
      "summary": "CONFIDENCE: MEDIUM"
    }
  ],
  "best_ticker": "MSFT",
  "analysis_count": 3,
  "processing_time_ms": 7367.0
}
```

---

## Strategy Optimizer

### `POST /optimize`

Grid-search the best strategy parameters for a ticker.

**Request body**
```json
{
  "ticker": "AAPL",
  "strategy": "macd",
  "optimize_for": "sharpe",
  "start_date": "2022-01-01",
  "end_date": "2024-12-31",
  "param_grid": {
    "fast": [8, 12, 16],
    "slow": [21, 26, 30],
    "signal_period": [7, 9, 12]
  }
}
```

| `strategy` | Valid `param_grid` keys |
|-----------|------------------------|
| `momentum` | `short_window`, `long_window` |
| `mean_reversion` | `window`, `z_threshold` |
| `rsi` | `period`, `oversold`, `overbought` |
| `macd` | `fast`, `slow`, `signal_period` |

| `optimize_for` | Description |
|---------------|-------------|
| `sharpe` | Maximize Sharpe ratio (default) |
| `return` | Maximize total return |
| `calmar` | Maximize Calmar ratio (return/drawdown) |

**Response `200`**
```json
{
  "ticker": "AAPL",
  "strategy": "macd",
  "optimize_for": "sharpe",
  "best_params": {"fast": 8, "slow": 30, "signal_period": 12},
  "best_score": 0.829,
  "default_score": 0.21,
  "improvement_pct": 295.0,
  "all_results": [...]
}
```

---

## Walk-Forward Validation

### `POST /walk-forward`

Validate strategy robustness using rolling train/test windows.

**Request body**
```json
{
  "ticker": "AAPL",
  "strategy": "macd",
  "train_period": "12mo",
  "test_period": "3mo",
  "step_period": "3mo",
  "params": {"fast": 12, "slow": 26, "signal_period": 9}
}
```

**Response `200`**
```json
{
  "ticker": "AAPL",
  "strategy": "macd",
  "in_sample_sharpe": 0.83,
  "out_of_sample_sharpe": 1.43,
  "robustness_ratio": 1.72,
  "verdict": "ROBUST",
  "windows": [
    {
      "window": 1,
      "train_start": "2022-01-01",
      "train_end": "2022-12-31",
      "test_start": "2023-01-01",
      "test_end": "2023-03-31",
      "train_sharpe": 0.21,
      "test_sharpe": 3.27,
      "best_params": {"fast": 12, "slow": 26, "signal_period": 9}
    }
  ],
  "processing_time_ms": 86.0
}
```

| `verdict` | `robustness_ratio` | Meaning |
|-----------|-------------------|---------|
| `ROBUST` | ≥ 0.7 | Strategy generalizes well |
| `MARGINAL` | 0.4–0.7 | Use with caution |
| `OVERFITTED` | < 0.4 | Strategy only works on training data |

---

## Portfolio

### `GET /portfolio/performance`

Get real-time P&L for all positions.

**Response `200`**
```json
{
  "positions": [
    {
      "ticker": "AAPL",
      "shares": 10,
      "entry_price": 175.0,
      "current_price": 291.58,
      "unrealized_pnl": 1165.8,
      "unrealized_pnl_pct": 66.6,
      "current_value": 2915.8,
      "cost_basis": 1750.0
    }
  ],
  "total_cost_basis": 1750.0,
  "total_current_value": 2915.8,
  "total_unrealized_pnl": 1165.8,
  "total_unrealized_pnl_pct": 66.6,
  "best_performer": "AAPL",
  "worst_performer": null,
  "position_count": 1,
  "last_updated": "2026-06-11T12:00:00"
}
```

### `POST /portfolio/positions`

Add a new position.

**Request body**
```json
{
  "ticker": "AAPL",
  "shares": 10,
  "entry_price": 175.0,
  "entry_date": "2023-01-15",
  "notes": "Long-term hold"
}
```

### `DELETE /portfolio/positions/{ticker}`

Remove a position by ticker.

---

## Earnings Calendar

### `GET /earnings/{ticker}`

Get next earnings date and risk warning for a single ticker.

**Response `200`**
```json
{
  "ticker": "AAPL",
  "next_earnings_date": "2026-07-31",
  "days_until_earnings": 49,
  "warning_level": "info",
  "message": "Next earnings: July 31, 2026 (49 days away)",
  "emoji": "ℹ️",
  "eps_estimate": null,
  "revenue_estimate": null,
  "has_upcoming_earnings": false
}
```

| `warning_level` | `days_until_earnings` | Action |
|----------------|----------------------|--------|
| `today` | 0 | 🚨 Do not trade |
| `critical` | 1–3 | ⚠️ HIGH RISK |
| `warning` | 4–7 | 📅 Be cautious |
| `info` | 8+ | ℹ️ Safe to analyze |

### `GET /earnings/calendar/upcoming`

Get earnings calendar for multiple tickers.

**Query params:** `tickers=AAPL,MSFT,GOOGL,NVDA`

---

## Ticker Search

### `GET /ticker/search`

Autocomplete ticker symbols via Yahoo Finance.

**Query params:** `q=AAPL`

**Response `200`**
```json
{
  "results": [
    {
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "exchange": "NMS",
      "type": "EQUITY",
      "type_display": "Equity"
    }
  ]
}
```

---

## WebSocket Endpoints

### `WS /alerts/ws`

Real-time price alerts. Connect with any WebSocket client.

**Client → Server messages:**
```json
// Add alert
{"action": "add", "ticker": "AAPL", "condition": "price_below", "threshold": 200.0, "message": "Buy opportunity"}

// Remove alert
{"action": "remove", "ticker": "AAPL"}

// List all alerts
{"action": "list"}

// Ping
{"action": "ping"}
```

| `condition` | Triggers when |
|------------|---------------|
| `price_below` | Current price < threshold |
| `price_above` | Current price > threshold |
| `change_pct_above` | Daily change % > threshold |
| `change_pct_below` | Daily change % < threshold |

**Server → Client messages:**
```json
// On connect
{"type": "connected", "message": "Connected. 0 alerts active.", "alerts": []}

// Alert fired
{"type": "alert", "ticker": "AAPL", "message": "🚨 AAPL dropped below $200", "current_price": 198.5, "threshold": 200.0}

// Price update (every 30s)
{"type": "price_update", "prices": {"AAPL": 291.58}, "timestamp": "2026-06-11T12:00:00"}
```

---

### `WS /live-chart/ws/{ticker}`

Real-time candlestick data streaming.

**URL params:** `ticker` — stock symbol (e.g. `AAPL`)  
**Query params:**
- `period` — `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y` (default: `1d`)
- `interval` — `1m`, `5m`, `15m`, `30m`, `60m`, `1d` (default: `1m`)

**Server → Client messages:**
```json
// Initial history (100 candles)
{
  "type": "history",
  "ticker": "AAPL",
  "candles": [
    {"time": 1718100000, "open": 290.5, "high": 292.1, "low": 289.8, "close": 291.6, "volume": 1234567}
  ]
}

// Live update (every 5s)
{
  "type": "update",
  "ticker": "AAPL",
  "candle": {"time": 1718100300, "open": 291.6, "high": 292.5, "low": 291.0, "close": 292.1, "volume": 234567}
}
```
