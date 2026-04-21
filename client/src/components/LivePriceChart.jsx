/**
 * LivePriceChart — Real-time candlestick chart with WebSocket streaming.
 *
 * Architecture:
 *   WebSocket ws://localhost:8000/live-chart/ws/{ticker}
 *     → server pushes OHLCV candle every 5 seconds
 *     → client maintains rolling 60-candle buffer
 *     → Recharts ComposedChart re-renders on each push
 *
 * Chart layers (ComposedChart):
 *   1. Custom CandlestickBar shape — green (bullish) / red (bearish) rectangles
 *   2. Line — EMA(20) computed client-side using O(1) incremental formula
 *   3. Second ComposedChart below — volume bars (green/red matching candle)
 *
 * EMA computed client-side:
 *   alpha = 2 / (span + 1)
 *   ema_t = alpha * close_t + (1 - alpha) * ema_{t-1}
 *   This is the same OnlineEMA algorithm from backend/engine/online_indicators.py
 *   replicated in JS — no need to send EMA from server.
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
const WS_BASE = "ws://localhost:8000/live-chart/ws";
const MAX_CANDLES = 60;
const EMA_SPAN = 20;
const EMA_ALPHA = 2 / (EMA_SPAN + 1);

const POPULAR_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META", "JPM"];

// ---------------------------------------------------------------------------
// EMA helper — O(1) incremental update (mirrors OnlineEMA in online_indicators.py)
// ---------------------------------------------------------------------------
function computeEMA(candles) {
  if (!candles.length) return [];
  const emas = new Array(candles.length).fill(null);
  let ema = candles[0].close; // seed with first close
  emas[0] = ema;
  for (let i = 1; i < candles.length; i++) {
    ema = EMA_ALPHA * candles[i].close + (1 - EMA_ALPHA) * ema;
    emas[i] = parseFloat(ema.toFixed(4));
  }
  return emas;
}

// ---------------------------------------------------------------------------
// Custom Candlestick Bar shape for Recharts
// Recharts has no native candlestick — we render it as a custom Bar shape.
// Each "bar" is actually a candlestick: body (open→close) + wick (low→high).
// ---------------------------------------------------------------------------
const CandlestickBar = (props) => {
  const { x, y, width, height, payload, yAxis } = props;
  if (!payload || !yAxis) return null;

  const { open, high, low, close } = payload;
  const isBullish = close >= open;
  const color = isBullish ? "#22c55e" : "#ef4444"; // green / red

  // yAxis.scale maps price → pixel y coordinate
  const scale = yAxis.scale;
  if (!scale) return null;

  const yHigh  = scale(high);
  const yLow   = scale(low);
  const yOpen  = scale(open);
  const yClose = scale(close);

  const bodyTop    = Math.min(yOpen, yClose);
  const bodyBottom = Math.max(yOpen, yClose);
  const bodyHeight = Math.max(bodyBottom - bodyTop, 1); // min 1px
  const centerX    = x + width / 2;
  const wickWidth  = 1.5;
  const bodyWidth  = Math.max(width * 0.6, 4);

  return (
    <g>
      {/* Wick — thin vertical line from low to high */}
      <line
        x1={centerX}
        y1={yHigh}
        x2={centerX}
        y2={yLow}
        stroke={color}
        strokeWidth={wickWidth}
      />
      {/* Body — rectangle from open to close */}
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
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-xs shadow-xl">
      <div className="text-gray-400 mb-1 font-mono">{d.time}</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
        <span className="text-gray-500">Open</span>
        <span className="text-white font-mono">${d.open?.toFixed(2)}</span>
        <span className="text-gray-500">High</span>
        <span className="text-green-400 font-mono">${d.high?.toFixed(2)}</span>
        <span className="text-gray-500">Low</span>
        <span className="text-red-400 font-mono">${d.low?.toFixed(2)}</span>
        <span className="text-gray-500">Close</span>
        <span className={`font-mono font-bold ${isBullish ? "text-green-400" : "text-red-400"}`}>
          ${d.close?.toFixed(2)}
        </span>
        <span className="text-gray-500">Volume</span>
        <span className="text-gray-300 font-mono">{(d.volume / 1e6).toFixed(2)}M</span>
        {d.ema != null && (
          <>
            <span className="text-gray-500">EMA(20)</span>
            <span className="text-blue-400 font-mono">${d.ema?.toFixed(2)}</span>
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
  const [candles, setCandles]         = useState([]);
  const [status, setStatus]           = useState("idle"); // idle | connecting | live | closed | error
  const [statusMsg, setStatusMsg]     = useState("");
  const [lastCandle, setLastCandle]   = useState(null);
  const [prevClose, setPrevClose]     = useState(null);

  const wsRef      = useRef(null);
  const emaRef     = useRef(null); // running EMA value
  const candlesRef = useRef([]);   // avoid stale closure in ws.onmessage

  // Keep candlesRef in sync
  useEffect(() => { candlesRef.current = candles; }, [candles]);

  // ---------------------------------------------------------------------------
  // WebSocket lifecycle
  // ---------------------------------------------------------------------------
  const connect = useCallback((sym) => {
    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setStatus("connecting");
    setStatusMsg(`Connecting to ${sym}...`);
    setCandles([]);
    candlesRef.current = [];
    emaRef.current = null;

    const ws = new WebSocket(`${WS_BASE}/${sym}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connecting");
      setStatusMsg("Waiting for data...");
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === "history") {
          // Initial 60-candle history
          const hist = msg.candles || [];
          // Compute EMA for all history candles
          const emas = computeEMA(hist);
          // Seed running EMA from last computed value
          emaRef.current = emas[emas.length - 1] ?? null;

          const enriched = hist.map((c, i) => ({ ...c, ema: emas[i] }));
          setCandles(enriched);
          candlesRef.current = enriched;

          if (enriched.length >= 2) {
            setPrevClose(enriched[enriched.length - 2].close);
          }
          setLastCandle(enriched[enriched.length - 1] ?? null);
          setStatus("live");
          setStatusMsg(`Streaming ${sym} — updates every 5s`);
        }

        else if (msg.type === "candle") {
          // Incremental update — update running EMA in O(1)
          const newEma = emaRef.current != null
            ? EMA_ALPHA * msg.close + (1 - EMA_ALPHA) * emaRef.current
            : msg.close;
          emaRef.current = newEma;

          const newCandle = {
            time:      msg.time,
            timestamp: msg.timestamp,
            open:      msg.open,
            high:      msg.high,
            low:       msg.low,
            close:     msg.close,
            volume:    msg.volume,
            ema:       parseFloat(newEma.toFixed(4)),
          };

          setCandles((prev) => {
            const updated = msg.is_new_candle
              ? [...prev, newCandle].slice(-MAX_CANDLES)   // new candle → append
              : [...prev.slice(0, -1), newCandle];          // same minute → replace last

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
      setStatus("error");
      setStatusMsg("WebSocket connection failed");
    };

    ws.onclose = () => {
      if (status !== "error") {
        setStatus("closed");
        setStatusMsg("Disconnected");
      }
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Disconnect on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------
  const handleStart = () => {
    const sym = inputTicker.trim().toUpperCase();
    if (!sym) return;
    setTicker(sym);
    connect(sym);
  };

  const handleStop = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus("idle");
    setStatusMsg("");
  };

  // ---------------------------------------------------------------------------
  // Derived values
  // ---------------------------------------------------------------------------
  const currentPrice  = lastCandle?.close ?? null;
  const priceChange   = currentPrice != null && prevClose != null
    ? currentPrice - prevClose : null;
  const changePct     = priceChange != null && prevClose
    ? (priceChange / prevClose) * 100 : null;
  const isPositive    = priceChange != null ? priceChange >= 0 : null;

  // Y-axis domain with 1% padding
  const prices = candles.flatMap((c) => [c.high, c.low]).filter(Boolean);
  const minP   = prices.length ? Math.min(...prices) : 0;
  const maxP   = prices.length ? Math.max(...prices) : 100;
  const pad    = (maxP - minP) * 0.01 || 1;
  const yDomain = [parseFloat((minP - pad).toFixed(2)), parseFloat((maxP + pad).toFixed(2))];

  const maxVol = candles.length ? Math.max(...candles.map((c) => c.volume || 0)) : 1;

  // Status badge config
  const statusConfig = {
    idle:       { color: "text-gray-500",  dot: "bg-gray-600",  label: "Idle" },
    connecting: { color: "text-yellow-400", dot: "bg-yellow-400 animate-pulse", label: "Connecting" },
    live:       { color: "text-green-400",  dot: "bg-green-400 animate-pulse",  label: "LIVE" },
    closed:     { color: "text-gray-400",   dot: "bg-gray-500",  label: "Disconnected" },
    error:      { color: "text-red-400",    dot: "bg-red-500",   label: "Error" },
  };
  const sc = statusConfig[status] || statusConfig.idle;

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="space-y-4">

      {/* ── Header ── */}
      <div className="card">
        <div className="flex flex-col sm:flex-row sm:items-end gap-4">
          {/* Ticker input */}
          <div className="flex-1">
            <label className="block text-xs text-gray-500 mb-1 uppercase tracking-wider">
              Ticker Symbol
            </label>
            <TickerAutocomplete
              value={inputTicker}
              onChange={setInputTicker}
              placeholder="AAPL, MSFT, TSLA..."
            />
          </div>

          {/* Controls */}
          <div className="flex gap-2">
            <button
              onClick={handleStart}
              disabled={status === "connecting"}
              className="px-5 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-900
                         text-white text-sm font-semibold rounded-lg transition-colors"
            >
              {status === "live" ? "🔄 Switch" : "▶ Start"}
            </button>
            {status === "live" && (
              <button
                onClick={handleStop}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white
                           text-sm font-semibold rounded-lg transition-colors"
              >
                ⏹ Stop
              </button>
            )}
          </div>
        </div>

        {/* Quick tickers */}
        <div className="flex flex-wrap gap-1.5 mt-3">
          {POPULAR_TICKERS.map((t) => (
            <button
              key={t}
              onClick={() => { setInputTicker(t); }}
              className={`text-xs px-2.5 py-1 rounded-full transition-colors ${
                inputTicker === t
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:text-gray-200"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* ── Price header ── */}
      {(status === "live" || status === "closed") && lastCandle && (
        <div className="card">
          <div className="flex items-center justify-between flex-wrap gap-4">
            {/* Ticker + price */}
            <div className="flex items-baseline gap-3">
              <span className="text-2xl font-black text-white">{ticker}</span>
              <span className="text-3xl font-bold text-white">
                ${currentPrice?.toFixed(2)}
              </span>
              {priceChange != null && (
                <span className={`text-lg font-semibold ${isPositive ? "text-green-400" : "text-red-400"}`}>
                  {isPositive ? "▲" : "▼"} {Math.abs(priceChange).toFixed(2)}
                  {" "}({isPositive ? "+" : ""}{changePct?.toFixed(2)}%)
                </span>
              )}
            </div>

            {/* Status badge */}
            <div className={`flex items-center gap-2 text-sm ${sc.color}`}>
              <span className={`w-2 h-2 rounded-full ${sc.dot}`}></span>
              <span className="font-medium">{sc.label}</span>
              {statusMsg && <span className="text-gray-500 text-xs">— {statusMsg}</span>}
            </div>
          </div>

          {/* OHLCV summary row */}
          {lastCandle && (
            <div className="flex flex-wrap gap-4 mt-3 text-sm">
              {[
                { label: "Open",   value: `$${lastCandle.open?.toFixed(2)}`,   color: "text-gray-300" },
                { label: "High",   value: `$${lastCandle.high?.toFixed(2)}`,   color: "text-green-400" },
                { label: "Low",    value: `$${lastCandle.low?.toFixed(2)}`,    color: "text-red-400" },
                { label: "Close",  value: `$${lastCandle.close?.toFixed(2)}`,  color: "text-white" },
                { label: "Volume", value: `${((lastCandle.volume || 0) / 1e6).toFixed(2)}M`, color: "text-gray-300" },
                { label: "EMA(20)", value: lastCandle.ema ? `$${lastCandle.ema?.toFixed(2)}` : "—", color: "text-blue-400" },
              ].map(({ label, value, color }) => (
                <div key={label} className="flex flex-col">
                  <span className="text-xs text-gray-500 uppercase tracking-wider">{label}</span>
                  <span className={`font-mono font-semibold ${color}`}>{value}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Idle state ── */}
      {status === "idle" && (
        <div className="card border-dashed border-gray-700 text-center py-16">
          <div className="text-5xl mb-4">📈</div>
          <h3 className="text-xl font-semibold text-gray-400 mb-2">Real-Time Candlestick Chart</h3>
          <p className="text-gray-500 text-sm max-w-sm mx-auto">
            Select a ticker and click <strong className="text-gray-300">▶ Start</strong> to stream
            live 1-minute OHLCV candles via WebSocket. Updates every 5 seconds.
          </p>
          <div className="mt-4 text-xs text-gray-600">
            EMA(20) overlay computed client-side using O(1) incremental algorithm
          </div>
        </div>
      )}

      {/* ── Connecting state ── */}
      {status === "connecting" && (
        <div className="card border border-yellow-800/50">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-full border-4 border-yellow-900 border-t-yellow-400 animate-spin"></div>
            <div>
              <h3 className="text-yellow-400 font-semibold">Connecting...</h3>
              <p className="text-gray-400 text-sm">Fetching historical candles for {ticker}</p>
            </div>
          </div>
        </div>
      )}

      {/* ── Error state ── */}
      {status === "error" && (
        <div className="card border border-red-800 bg-red-900/20">
          <div className="flex items-start gap-3">
            <span className="text-2xl">❌</span>
            <div>
              <h3 className="text-red-400 font-semibold">Connection Failed</h3>
              <p className="text-red-300/80 text-sm mt-1">{statusMsg}</p>
              <button
                onClick={handleStart}
                className="mt-2 text-xs text-red-400 hover:text-red-300 underline"
              >
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
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
              {ticker} — 1m Candlestick
            </h3>
            <div className="flex items-center gap-4 text-xs text-gray-500">
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

          <ResponsiveContainer width="100%" height={320}>
            <ComposedChart
              data={candles}
              margin={{ top: 8, right: 16, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
              <XAxis
                dataKey="time"
                tick={{ fill: "#6b7280", fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                interval={Math.floor(candles.length / 8)}
              />
              <YAxis
                domain={yDomain}
                tick={{ fill: "#6b7280", fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => `$${v.toFixed(0)}`}
                width={55}
              />
              <Tooltip content={<CandleTooltip />} />

              {/* Candlestick bars — rendered as custom shape */}
              <Bar
                dataKey="high"
                shape={<CandlestickBar />}
                isAnimationActive={false}
              />

              {/* EMA(20) line overlay */}
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
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Volume
          </h3>
          <ResponsiveContainer width="100%" height={100}>
            <ComposedChart
              data={candles}
              margin={{ top: 0, right: 16, left: 0, bottom: 0 }}
            >
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
                  <Cell
                    key={i}
                    fill={c.close >= c.open ? "#16a34a" : "#dc2626"}
                    opacity={0.7}
                  />
                ))}
              </Bar>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Info footer ── */}
      <div className="text-xs text-gray-600 text-center">
        Data: yfinance 1-minute bars · WebSocket push every 5s ·
        EMA(20) computed client-side (O(1) incremental) · Max {MAX_CANDLES} candles displayed
      </div>
    </div>
  );
};

export default LivePriceChart;
