/**
 * LivePriceChart — Real-time candlestick chart with WebSocket streaming.
 *
 * Features:
 *   - Time range selector: Today / 5 Days / 1 Month / 3 Months / 6 Months / 1 Year
 *   - Interval selector: 1m / 5m / 15m / 30m / 1h / 1D (auto-filtered per range)
 *   - Custom CandlestickBar SVG shape (Recharts ComposedChart)
 *   - EMA(20) line overlay computed client-side O(1) incremental
 *   - Volume bars (green/red matching candle color)
 *   - Rolling 100-candle buffer via useRef
 *   - WebSocket lifecycle: connect/disconnect/error/heartbeat
 *
 * WebSocket URL format:
 *   ws://localhost:8000/live-chart/ws/{ticker}?period={period}&interval={interval}
 *
 * Period → valid intervals (mirrors backend VALID_COMBOS):
 *   1d  → 1m, 5m, 15m, 30m, 1h
 *   5d  → 1m, 5m, 15m, 30m, 1h
 *   1mo → 5m, 15m, 30m, 1h, 1D
 *   3mo → 15m, 30m, 1h, 1D
 *   6mo → 1D, 1W
 *   1y  → 1D, 1W
 */

import { useState, useEffect, useRef, useCallback } from "react";
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import TickerAutocomplete from "./TickerAutocomplete";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
// Derive WebSocket base URL from current page host so it works on any port
// WebSocket URL resolution (works for dev, prod Vercel→Render, and Docker):
// 1. VITE_WS_URL env var (set in Vercel env or .env.local) — highest priority
// 2. VITE_API_URL starting with http → swap protocol to ws
// 3. Fall back to same host as the page (works when backend is co-hosted)
const _wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
const _apiBase = import.meta.env.VITE_API_URL || "";
const _wsExplicit = import.meta.env.VITE_WS_URL || "";
const WS_BASE = _wsExplicit
  ? _wsExplicit.replace(/\/?$/, "") + "/live-chart/ws"
  : _apiBase.startsWith("http")
  ? _apiBase.replace(/^http/, "ws") + "/live-chart/ws"
  : `${_wsProto}//${window.location.hostname}:8000/live-chart/ws`;

// HTTP health URL for wake-up ping (Render free tier cold start fix)
const HTTP_HEALTH_URL = _wsExplicit
  ? _wsExplicit.replace(/^wss?/, "https").replace(/\/?$/, "") + "/health"
  : _apiBase.startsWith("http")
  ? _apiBase.replace(/\/?$/, "") + "/health"
  : `${window.location.protocol}//${window.location.hostname}:8000/health`;

const MAX_CANDLES = 100;
const WS_MAX_RETRIES = 5;
const WS_BASE_DELAY_MS = 2000;
const EMA_SPAN   = 20;
const EMA_ALPHA  = 2 / (EMA_SPAN + 1);

const POPULAR_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META", "JPM"];

// Period options with labels and valid intervals
const PERIOD_OPTIONS = [
  { value: "1d",  label: "Today",   intervals: ["1m", "5m", "15m", "30m", "60m"] },
  { value: "5d",  label: "5 Days",  intervals: ["1m", "5m", "15m", "30m", "60m"] },
  { value: "1mo", label: "1 Month", intervals: ["5m", "15m", "30m", "60m", "1d"] },
  { value: "3mo", label: "3 Months",intervals: ["15m", "30m", "60m", "1d"] },
  { value: "6mo", label: "6 Months",intervals: ["1d", "1wk"] },
  { value: "1y",  label: "1 Year",  intervals: ["1d", "1wk"] },
];

// Interval display labels
const INTERVAL_LABELS = {
  "1m":  "1m",
  "5m":  "5m",
  "15m": "15m",
  "30m": "30m",
  "60m": "1h",
  "1d":  "1D",
  "1wk": "1W",
};

// ---------------------------------------------------------------------------
// EMA helper — O(1) incremental (mirrors OnlineEMA in online_indicators.py)
// ---------------------------------------------------------------------------
function computeEMA(candles) {
  if (!candles.length) return [];
  const emas = new Array(candles.length).fill(null);
  let ema = candles[0].close;
  emas[0] = ema;
  for (let i = 1; i < candles.length; i++) {
    ema = EMA_ALPHA * candles[i].close + (1 - EMA_ALPHA) * ema;
    emas[i] = parseFloat(ema.toFixed(4));
  }
  return emas;
}

// ---------------------------------------------------------------------------
// Custom Candlestick Bar shape for Recharts
// ---------------------------------------------------------------------------
const CandlestickBar = (props) => {
  const { x, y, width, height, payload, yAxis } = props;
  if (!payload || !yAxis) return null;

  const { open, high, low, close } = payload;
  const isBullish = close >= open;
  const color     = isBullish ? "#22c55e" : "#ef4444";
  const scale     = yAxis.scale;
  if (!scale) return null;

  const yHigh      = scale(high);
  const yLow       = scale(low);
  const yOpen      = scale(open);
  const yClose     = scale(close);
  const bodyTop    = Math.min(yOpen, yClose);
  const bodyBottom = Math.max(yOpen, yClose);
  const bodyHeight = Math.max(bodyBottom - bodyTop, 1);
  const centerX    = x + width / 2;
  const bodyWidth  = Math.max(width * 0.6, 3);

  return (
    <g>
      <line x1={centerX} y1={yHigh} x2={centerX} y2={yLow} stroke={color} strokeWidth={1.5} />
      <rect
        x={centerX - bodyWidth / 2}
        y={bodyTop}
        width={bodyWidth}
        height={bodyHeight}
        fill={color}
        stroke={color}
        strokeWidth={0.5}
        opacity={0.9}
      />
    </g>
  );
};

// ---------------------------------------------------------------------------
// Custom Tooltip
// ---------------------------------------------------------------------------
const CandleTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  const isBullish = d.close >= d.open;
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-3 text-xs shadow-xl">
      <div className="text-slate-400 mb-1 font-mono">{d.time}</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
        <span className="text-slate-400">Open</span>
        <span className="text-slate-900 font-mono">${d.open?.toFixed(2)}</span>
        <span className="text-slate-400">High</span>
        <span className="text-emerald-600 font-mono">${d.high?.toFixed(2)}</span>
        <span className="text-slate-400">Low</span>
        <span className="text-red-400 font-mono">${d.low?.toFixed(2)}</span>
        <span className="text-slate-400">Close</span>
        <span className={`font-mono font-bold ${isBullish ? "text-emerald-600" : "text-red-400"}`}>
          ${d.close?.toFixed(2)}
        </span>
        <span className="text-slate-400">Volume</span>
        <span className="text-slate-600 font-mono">{(d.volume / 1e6).toFixed(2)}M</span>
        {d.ema != null && (
          <>
            <span className="text-slate-400">EMA(20)</span>
            <span className="text-indigo-500 font-mono">${d.ema?.toFixed(2)}</span>
          </>
        )}
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
const LivePriceChart = () => {
  const [ticker, setTicker]           = useState("AAPL");
  const [inputTicker, setInputTicker] = useState("AAPL");
  const [period, setPeriod]           = useState("1d");
  const [interval, setInterval]       = useState("1m");
  const [candles, setCandles]         = useState([]);
  const [status, setStatus]           = useState("idle");
  const [statusMsg, setStatusMsg]     = useState("");
  const [lastCandle, setLastCandle]   = useState(null);
  const [prevClose, setPrevClose]     = useState(null);

  const wsRef         = useRef(null);
  const emaRef        = useRef(null);
  const candlesRef    = useRef([]);
  const retryRef      = useRef(0);
  const retryTimerRef = useRef(null);
  const activeSymRef  = useRef(null);
  const activePerRef  = useRef(null);
  const activeIntvRef = useRef(null);

  useEffect(() => { candlesRef.current = candles; }, [candles]);

  // Current period config
  const periodConfig = PERIOD_OPTIONS.find((p) => p.value === period) || PERIOD_OPTIONS[0];

  // ---------------------------------------------------------------------------
  // WebSocket lifecycle
  // ---------------------------------------------------------------------------
  const connect = useCallback((sym, per, intv, isRetry = false) => {
    if (wsRef.current) {
      wsRef.current.onclose = null; // prevent retry loop on manual reconnect
      wsRef.current.close();
      wsRef.current = null;
    }
    clearTimeout(retryTimerRef.current);

    // Store active params so retry can reuse them
    activeSymRef.current  = sym;
    activePerRef.current  = per;
    activeIntvRef.current = intv;

    if (!isRetry) {
      retryRef.current = 0;
      setCandles([]);
      candlesRef.current = [];
      emaRef.current = null;
    }

    setStatus("connecting");
    setStatusMsg(isRetry ? `Reconnecting to ${sym}… (attempt ${retryRef.current}/${WS_MAX_RETRIES})` : `Connecting to ${sym}...`);

    const url = `${WS_BASE}/${sym}?period=${per}&interval=${intv}`;
    const ws  = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      retryRef.current = 0; // reset on successful connect
      setStatus("connecting");
      setStatusMsg("Waiting for data...");
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === "history") {
          const hist = msg.candles || [];
          if (hist.length === 0) {
            // Market closed or no data — show informative message, stay "live"
            setStatus("live");
            setStatusMsg(`${sym} · Market closed or no data for this range`);
            return;
          }
          const emas  = computeEMA(hist);
          emaRef.current = emas[emas.length - 1] ?? null;
          const enriched = hist.map((c, i) => ({ ...c, ema: emas[i] }));
          setCandles(enriched);
          candlesRef.current = enriched;
          if (enriched.length >= 2) setPrevClose(enriched[enriched.length - 2].close);
          setLastCandle(enriched[enriched.length - 1] ?? null);
          setStatus("live");
          setStatusMsg(`${sym} · ${PERIOD_OPTIONS.find(p=>p.value===per)?.label} · ${INTERVAL_LABELS[intv] || intv}`);
        }

        else if (msg.type === "candle") {
          const newEma = emaRef.current != null
            ? EMA_ALPHA * msg.close + (1 - EMA_ALPHA) * emaRef.current
            : msg.close;
          emaRef.current = newEma;

          const newCandle = {
            time: msg.time, timestamp: msg.timestamp,
            open: msg.open, high: msg.high, low: msg.low, close: msg.close,
            volume: msg.volume, ema: parseFloat(newEma.toFixed(4)),
          };

          setCandles((prev) => {
            const updated = msg.is_new_candle
              ? [...prev, newCandle].slice(-MAX_CANDLES)
              : [...prev.slice(0, -1), newCandle];
            candlesRef.current = updated;
            return updated;
          });
          setLastCandle(newCandle);
          setStatus("live");
        }

        else if (msg.type === "heartbeat") {
          setStatusMsg(msg.message || "Market may be closed");
        }

        else if (msg.type === "error") {
          setStatus("error");
          setStatusMsg(msg.message || "Stream error");
        }

      } catch (e) {
        console.error("LiveChart WS parse error:", e);
      }
    };

    ws.onerror = () => {
      // onerror always precedes onclose — let onclose handle retry
    };

    ws.onclose = () => {
      setStatus((prev) => {
        if (prev === "idle") return "idle"; // user stopped manually
        if (prev === "error") return "error";
        // Auto-retry on unexpected close
        if (retryRef.current < WS_MAX_RETRIES && activeSymRef.current) {
          retryRef.current += 1;
          const delay = WS_BASE_DELAY_MS * Math.pow(2, retryRef.current - 1);
          retryTimerRef.current = setTimeout(() => {
            connect(activeSymRef.current, activePerRef.current, activeIntvRef.current, true);
          }, delay);
          return "connecting";
        }
        return "closed";
      });
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup on unmount — also cancel any pending retry timer
  useEffect(() => {
    return () => {
      clearTimeout(retryTimerRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null; // prevent retry after unmount
        wsRef.current.close();
      }
    };
  }, []);

  // Track whether chart is currently streaming (via ref to avoid stale closure)
  const isStreamingRef = useRef(false);
  useEffect(() => {
    isStreamingRef.current = (status === "live" || status === "connecting");
  }, [status]);

  // Auto-reconnect when period OR interval changes — only if already streaming
  useEffect(() => {
    if (isStreamingRef.current && ticker) {
      connect(ticker, period, interval);
    }
  }, [period, interval]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------
  const handleStart = () => {
    const sym = inputTicker.trim().toUpperCase();
    if (!sym) return;
    setTicker(sym);
    // Fire-and-forget wake-up ping to warm Render free tier before WS connect
    fetch(HTTP_HEALTH_URL, { method: "GET", signal: AbortSignal.timeout(5000) })
      .catch(() => {}) // ignore — just warming up
      .finally(() => connect(sym, period, interval));
  };

  const handleStop = () => {
    clearTimeout(retryTimerRef.current);
    activeSymRef.current = null; // prevent retry after manual stop
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus("idle");
    setStatusMsg("");
    isStreamingRef.current = false;
  };

  // When period changes: auto-select finest valid interval,
  // then the useEffect above fires and auto-reconnects if streaming
  const handlePeriodChange = (newPeriod) => {
    const cfg = PERIOD_OPTIONS.find((p) => p.value === newPeriod);
    const newInterval = cfg?.intervals[0] || "1d";
    setPeriod(newPeriod);
    setInterval(newInterval);
    // useEffect([period, interval]) will auto-reconnect
  };

  // ---------------------------------------------------------------------------
  // Derived values
  // ---------------------------------------------------------------------------
  const currentPrice = lastCandle?.close ?? null;
  const priceChange  = currentPrice != null && prevClose != null ? currentPrice - prevClose : null;
  const changePct    = priceChange != null && prevClose ? (priceChange / prevClose) * 100 : null;
  const isPositive   = priceChange != null ? priceChange >= 0 : null;

  const prices  = candles.flatMap((c) => [c.high, c.low]).filter(Boolean);
  const minP    = prices.length ? Math.min(...prices) : 0;
  const maxP    = prices.length ? Math.max(...prices) : 100;
  const pad     = (maxP - minP) * 0.01 || 1;
  const yDomain = [parseFloat((minP - pad).toFixed(2)), parseFloat((maxP + pad).toFixed(2))];

  // X-axis tick interval — show ~8 labels regardless of candle count
  const xTickInterval = Math.max(1, Math.floor(candles.length / 8));

  const statusConfig = {
    idle:       { color: "text-slate-400",   dot: "bg-slate-300",                    label: "Idle" },
    connecting: { color: "text-amber-500", dot: "bg-amber-500 animate-pulse",    label: "Connecting" },
    live:       { color: "text-emerald-600",  dot: "bg-emerald-500 animate-pulse",     label: "LIVE" },
    closed:     { color: "text-slate-400",   dot: "bg-gray-500",                    label: "Disconnected" },
    error:      { color: "text-red-400",    dot: "bg-red-500",                     label: "Error" },
  };
  const sc = statusConfig[status] || statusConfig.idle;

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="space-y-4">

      {/* ── Controls card ── */}
      <div className="card space-y-4">

        {/* Row 1: Ticker + Start/Stop */}
        <div className="flex flex-col sm:flex-row sm:items-end gap-3">
          <div className="flex-1">
            <label className="block text-xs text-slate-400 mb-1 uppercase tracking-wider">Ticker</label>
            <TickerAutocomplete value={inputTicker} onChange={setInputTicker} placeholder="AAPL, MSFT, TSLA..." />
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleStart}
              disabled={status === "connecting"}
              className="btn-primary text-sm"
            >
              {status === "live" ? "🔄 Reload" : "▶ Start"}
            </button>
            {(status === "live" || status === "connecting") && (
              <button
                onClick={handleStop}
                className="px-4 py-2 bg-slate-200 hover:bg-slate-300 text-slate-900
                           text-sm font-semibold rounded-lg transition-colors"
              >
                ⏹ Stop
              </button>
            )}
          </div>
        </div>

        {/* Row 2: Quick tickers */}
        <div className="flex flex-wrap gap-1.5">
          {POPULAR_TICKERS.map((t) => (
            <button
              key={t}
              onClick={() => setInputTicker(t)}
              className={`text-xs px-2.5 py-1 rounded-full font-medium transition-colors ${
                inputTicker === t
                  ? "bg-indigo-600 text-white"
                  : "bg-slate-100 text-slate-500 hover:text-slate-700 hover:bg-slate-200"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        {/* Row 3: Period selector */}
        <div>
          <label className="block text-xs text-slate-400 mb-1.5 uppercase tracking-wider">Time Range</label>
          <div className="flex flex-wrap gap-1.5">
            {PERIOD_OPTIONS.map((p) => (
              <button
                key={p.value}
                onClick={() => handlePeriodChange(p.value)}
                className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-colors ${
                  period === p.value
                    ? "bg-indigo-600 text-white"
                    : "bg-slate-100 text-slate-500 hover:text-slate-700 hover:bg-slate-200"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* Row 4: Interval selector (filtered by period) */}
        <div>
          <label className="block text-xs text-slate-400 mb-1.5 uppercase tracking-wider">Interval</label>
          <div className="flex flex-wrap gap-1.5">
            {periodConfig.intervals.map((intv) => (
              <button
                key={intv}
                onClick={() => setInterval(intv)}
                className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-colors ${
                  interval === intv
                    ? "bg-indigo-600 text-white"
                    : "bg-slate-100 text-slate-500 hover:text-slate-700 hover:bg-slate-200"
                }`}
              >
                {INTERVAL_LABELS[intv] || intv}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Price header ── */}
      {(status === "live" || status === "closed") && lastCandle && (
        <div className="card">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-baseline gap-3">
              <span className="text-2xl font-black text-slate-900">{ticker}</span>
              <span className="text-3xl font-bold text-slate-900">${currentPrice?.toFixed(2)}</span>
              {priceChange != null && (
                <span className={`text-lg font-semibold ${isPositive ? "text-emerald-600" : "text-red-400"}`}>
                  {isPositive ? "▲" : "▼"} {Math.abs(priceChange).toFixed(2)}
                  {" "}({isPositive ? "+" : ""}{changePct?.toFixed(2)}%)
                </span>
              )}
            </div>
            <div className={`flex items-center gap-2 text-sm ${sc.color}`}>
              <span className={`w-2 h-2 rounded-full ${sc.dot}`}></span>
              <span className="font-medium">{sc.label}</span>
              {statusMsg && <span className="text-slate-400 text-xs">— {statusMsg}</span>}
            </div>
          </div>

          {/* OHLCV row */}
          <div className="flex flex-wrap gap-4 mt-3 text-sm">
            {[
              { label: "Open",    value: `$${lastCandle.open?.toFixed(2)}`,  color: "text-slate-600" },
              { label: "High",    value: `$${lastCandle.high?.toFixed(2)}`,  color: "text-emerald-600" },
              { label: "Low",     value: `$${lastCandle.low?.toFixed(2)}`,   color: "text-red-400" },
              { label: "Close",   value: `$${lastCandle.close?.toFixed(2)}`, color: "text-slate-900" },
              { label: "Volume",  value: `${((lastCandle.volume||0)/1e6).toFixed(2)}M`, color: "text-slate-600" },
              { label: "EMA(20)", value: lastCandle.ema ? `$${lastCandle.ema?.toFixed(2)}` : "—", color: "text-indigo-500" },
            ].map(({ label, value, color }) => (
              <div key={label} className="flex flex-col">
                <span className="text-xs text-slate-400 uppercase tracking-wider">{label}</span>
                <span className={`font-mono font-semibold ${color}`}>{value}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Idle state ── */}
      {status === "idle" && (
        <div className="card border-dashed border-slate-200 text-center py-16">
          <div className="text-5xl mb-4">📈</div>
          <h3 className="text-xl font-semibold text-slate-400 mb-2">Real-Time Candlestick Chart</h3>
          <p className="text-slate-400 text-sm max-w-sm mx-auto">
            Select a ticker, time range, and interval — then click <strong className="text-slate-600">▶ Start</strong>.
          </p>
          <div className="mt-3 text-xs text-slate-400">
            WebSocket streaming · EMA(20) O(1) incremental · Up to 100 candles
          </div>
        </div>
      )}

      {/* ── Connecting / Reconnecting ── */}
      {status === "connecting" && (
        <div className="card border border-amber-200">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-full border-4 border-yellow-900 border-t-yellow-400 animate-spin flex-shrink-0"></div>
            <div>
              <h3 className="text-amber-500 font-semibold">{statusMsg.includes("Reconnecting") ? "Reconnecting…" : "Connecting…"}</h3>
              <p className="text-slate-400 text-sm">{statusMsg}</p>
              {statusMsg.includes("Reconnecting") && (
                <p className="text-slate-400 text-xs mt-1">
                  Backend may be waking up (Render free tier cold start ~30s)
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Live but no candles (market closed) ── */}
      {status === "live" && candles.length === 0 && (
        <div className="card border border-slate-200 text-center py-12">
          <div className="text-4xl mb-3">🌙</div>
          <h3 className="text-lg font-semibold text-slate-500 mb-1">Market Closed</h3>
          <p className="text-slate-400 text-sm max-w-sm mx-auto">
            No data available for <strong className="text-slate-600">{ticker}</strong> in the selected range.
            Try a longer time range (e.g. <strong>1 Month</strong>) or check back during market hours
            (Mon–Fri 9:30 AM – 4:00 PM ET).
          </p>
          <p className="text-slate-400 text-xs mt-3">{statusMsg}</p>
        </div>
      )}

      {/* ── Error ── */}
      {status === "error" && (
        <div className="card border border-red-800 bg-red-900/20">
          <div className="flex items-start gap-3">
            <span className="text-2xl">❌</span>
            <div>
              <h3 className="text-red-400 font-semibold">Connection Failed</h3>
              <p className="text-red-300/80 text-sm mt-1">{statusMsg}</p>
              <button onClick={handleStart} className="mt-2 text-xs text-red-400 hover:text-red-300 underline">
                Retry
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Candlestick Chart ── */}
      {candles.length > 0 && (
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
              {ticker} · {PERIOD_OPTIONS.find(p=>p.value===period)?.label} · {INTERVAL_LABELS[interval]||interval} Candlestick
            </h3>
            <div className="flex items-center gap-4 text-xs text-slate-400">
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-blue-400 inline-block"></span> EMA(20)
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 bg-green-500 inline-block rounded-sm"></span> Bullish
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 bg-red-500 inline-block rounded-sm"></span> Bearish
              </span>
            </div>
          </div>

          <ResponsiveContainer width="100%" height={340}>
            <ComposedChart data={candles} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
              <XAxis
                dataKey="time"
                tick={{ fill: "#6b7280", fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                interval={xTickInterval}
              />
              <YAxis
                domain={yDomain}
                tick={{ fill: "#6b7280", fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => `$${v.toFixed(0)}`}
                width={58}
              />
              <Tooltip content={<CandleTooltip />} />
              <Bar dataKey="high" shape={<CandlestickBar />} isAnimationActive={false} />
              <Line
                type="monotone"
                dataKey="ema"
                stroke="#60a5fa"
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Volume Chart ── */}
      {candles.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Volume</h3>
          <ResponsiveContainer width="100%" height={100}>
            <ComposedChart data={candles} margin={{ top: 0, right: 16, left: 0, bottom: 0 }}>
              <XAxis dataKey="time" hide />
              <YAxis
                tick={{ fill: "#6b7280", fontSize: 9 }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => `${(v / 1e6).toFixed(1)}M`}
                width={45}
              />
              <Tooltip
                formatter={(v) => [`${(v / 1e6).toFixed(2)}M`, "Volume"]}
                contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
                labelStyle={{ color: "#9ca3af" }}
                itemStyle={{ color: "#d1d5db" }}
              />
              <Bar dataKey="volume" isAnimationActive={false} maxBarSize={12}>
                {candles.map((c, i) => (
                  <Cell key={i} fill={c.close >= c.open ? "#16a34a" : "#dc2626"} opacity={0.7} />
                ))}
              </Bar>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Footer ── */}
      <div className="text-xs text-slate-400 text-center">
        Data: yfinance · WebSocket push every 5s (intraday) / 60s (daily) ·
        EMA(20) O(1) incremental · Max {MAX_CANDLES} candles
      </div>
    </div>
  );
};

export default LivePriceChart;
