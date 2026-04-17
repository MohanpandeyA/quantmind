/**
 * CompareStocks — compare multiple tickers and rank them.
 *
 * WHY THIS COMPONENT EXISTS:
 *   "Which of AAPL, MSFT, NVDA, GOOGL should I buy?" is a common question.
 *   This component runs all analyses in parallel and shows a ranked table
 *   with composite scores, making the decision obvious at a glance.
 *
 * RANKING FORMULA (shown to user for transparency):
 *   score = 0.4 × Sharpe + 0.3 × Return + 0.2 × (1-MDD) + 0.1 × WinRate
 */

import { useState } from "react";
import axios from "axios";
import TickerAutocomplete from "./TickerAutocomplete";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

const SIGNAL_COLORS = {
  BUY: "text-green-400 bg-green-900/30",
  SELL: "text-red-400 bg-red-900/30",
  HOLD: "text-yellow-400 bg-yellow-900/30",
};

const CompareStocks = () => {
  const [tickers, setTickers] = useState("AAPL,MSFT,NVDA,GOOGL,JPM");
  const [addTicker, setAddTicker] = useState("");
  const [query, setQuery] = useState("Which is the best investment right now?");
  const [startDate, setStartDate] = useState("2022-01-01");
  const [endDate, setEndDate] = useState("2024-12-31");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const appendTicker = (symbol) => {
    const list = tickers.split(",").map((t) => t.trim()).filter(Boolean);
    if (!list.includes(symbol)) {
      setTickers([...list, symbol].join(","));
    }
    setAddTicker("");
  };

  const runComparison = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults(null);

    const tickerList = tickers.split(",").map((t) => t.trim().toUpperCase()).filter(Boolean);
    if (tickerList.length < 2) {
      setError("Enter at least 2 tickers separated by commas");
      setLoading(false);
      return;
    }

    try {
      const resp = await axios.post(`${BASE_URL}/compare`, {
        tickers: tickerList,
        query,
        start_date: startDate,
        end_date: endDate,
      });
      setResults(resp.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const fmt = {
    pct: (v) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`,
    score: (v) => (v * 100).toFixed(1),
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white flex items-center gap-2">
        <span>📊</span> Compare Stocks
      </h2>

      <div className="card">
        <form onSubmit={runComparison} className="space-y-3">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Tickers (comma-separated)</label>
            <input
              value={tickers}
              onChange={(e) => setTickers(e.target.value)}
              placeholder="AAPL,MSFT,NVDA,GOOGL,JPM"
              className="input-field"
              disabled={loading}
            />
            <div className="flex gap-2 mt-2">
              <TickerAutocomplete
                value={addTicker}
                onChange={setAddTicker}
                onSelect={({ symbol }) => appendTicker(symbol)}
                placeholder="Search to add a ticker..."
                disabled={loading}
                showHint={false}
                className="flex-1"
              />
              <button
                type="button"
                onClick={() => addTicker.trim() && appendTicker(addTicker.trim().toUpperCase())}
                disabled={!addTicker.trim() || loading}
                className="text-xs text-blue-400 hover:text-blue-300 bg-blue-900/30 px-3 py-1.5 rounded-lg disabled:opacity-40"
              >
                + Add
              </button>
            </div>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Question</label>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="input-field"
              disabled={loading}
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Start Date</label>
              <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="input-field text-sm" disabled={loading} />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">End Date</label>
              <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="input-field text-sm" disabled={loading} />
            </div>
          </div>
          <button type="submit" disabled={loading} className="btn-primary w-full flex items-center justify-center gap-2">
            {loading ? (
              <><svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg> Comparing (parallel)...</>
            ) : "⚡ Compare All"}
          </button>
        </form>
      </div>

      {error && <div className="card border border-red-800 bg-red-900/20 text-red-400 text-sm">{error}</div>}

      {results && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-300">
              Rankings — {results.total_tickers} tickers
            </h3>
            <div className="text-xs text-gray-500">
              {(results.processing_time_ms / 1000).toFixed(1)}s | Score = 0.4×Sharpe + 0.3×Return + 0.2×(1-MDD) + 0.1×WinRate
            </div>
          </div>

          <div className="space-y-2">
            {results.rankings.map((r) => (
              <div
                key={r.ticker}
                className={`flex items-center gap-4 p-3 rounded-lg border ${
                  r.rank === 1 ? "border-yellow-700 bg-yellow-900/10" : "border-gray-800 bg-gray-800/30"
                }`}
              >
                {/* Rank */}
                <div className={`text-2xl font-black w-8 text-center ${r.rank === 1 ? "text-yellow-400" : "text-gray-500"}`}>
                  {r.rank === 1 ? "🥇" : r.rank === 2 ? "🥈" : r.rank === 3 ? "🥉" : `#${r.rank}`}
                </div>

                {/* Ticker + Signal */}
                <div className="w-24">
                  <div className="text-white font-bold font-mono text-lg">{r.ticker}</div>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${SIGNAL_COLORS[r.signal] || SIGNAL_COLORS.HOLD}`}>
                    {r.signal}
                  </span>
                </div>

                {/* Score bar */}
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs text-gray-400">Score</span>
                    <span className="text-sm font-bold text-white">{fmt.score(r.composite_score)}</span>
                  </div>
                  <div className="bg-gray-800 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${Math.min(100, r.composite_score * 100)}%` }}
                    />
                  </div>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-3 gap-3 text-right text-sm">
                  <div>
                    <div className="text-gray-400 text-xs">Sharpe</div>
                    <div className={r.sharpe_ratio >= 1 ? "text-green-400" : r.sharpe_ratio >= 0.5 ? "text-yellow-400" : "text-red-400"}>
                      {r.sharpe_ratio.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400 text-xs">Return</div>
                    <div className={r.total_return >= 0 ? "text-green-400" : "text-red-400"}>
                      {fmt.pct(r.total_return)}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400 text-xs">MDD</div>
                    <div className={r.max_drawdown <= 0.15 ? "text-green-400" : r.max_drawdown <= 0.25 ? "text-yellow-400" : "text-red-400"}>
                      {fmt.pct(r.max_drawdown)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 pt-4 border-t border-gray-800 text-sm text-gray-400">
            🏆 Best: <span className="text-green-400 font-semibold">{results.best_ticker}</span>
            {" | "}
            📉 Worst: <span className="text-red-400 font-semibold">{results.worst_ticker}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default CompareStocks;
