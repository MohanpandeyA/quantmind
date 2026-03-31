/**
 * StrategyOptimizer — find the best strategy parameters via grid search.
 *
 * WHY THIS COMPONENT EXISTS:
 *   Default parameters (short=20, long=50) are generic. AAPL's optimal
 *   parameters might be (10, 30) giving Sharpe=1.42 vs default Sharpe=0.38.
 *   This component shows the trader exactly which parameters work best
 *   for their specific ticker, with a visual comparison table.
 *
 * TRANSPARENCY: Shows ALL tested combinations, not just the best.
 *   Traders can see the full picture and make informed decisions.
 */

import { useState } from "react";
import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

const StrategyOptimizer = () => {
  const [ticker, setTicker] = useState("AAPL");
  const [strategy, setStrategy] = useState("momentum");
  const [optimizeFor, setOptimizeFor] = useState("sharpe");
  const [startDate, setStartDate] = useState("2022-01-01");
  const [endDate, setEndDate] = useState("2024-12-31");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runOptimization = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const resp = await axios.post(`${BASE_URL}/optimize`, {
        ticker: ticker.toUpperCase(),
        strategy,
        optimize_for: optimizeFor,
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
    num: (v) => v?.toFixed(3) ?? "—",
  };

  const metricLabel = {
    sharpe: "Sharpe Ratio",
    total_return: "Total Return",
    calmar: "Calmar Ratio",
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white flex items-center gap-2">
        <span>⚙️</span> Strategy Optimizer
      </h2>

      <div className="card">
        <p className="text-sm text-gray-400 mb-4">
          Tests all parameter combinations and finds the best for your ticker.
          Uses data cache — only 1 yfinance download for all combinations.
        </p>
        <form onSubmit={runOptimization} className="space-y-3">
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Ticker</label>
              <input
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                className="input-field"
                disabled={loading}
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Strategy</label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="input-field"
                disabled={loading}
              >
                <option value="momentum">Momentum (MA Crossover)</option>
                <option value="mean_reversion">Mean Reversion (Bollinger)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Optimize For</label>
              <select
                value={optimizeFor}
                onChange={(e) => setOptimizeFor(e.target.value)}
                className="input-field"
                disabled={loading}
              >
                <option value="sharpe">Sharpe Ratio (recommended)</option>
                <option value="total_return">Total Return</option>
                <option value="calmar">Calmar Ratio</option>
              </select>
            </div>
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
              <><svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/></svg> Running grid search...</>
            ) : "🔍 Find Best Parameters"}
          </button>
        </form>
      </div>

      {error && <div className="card border border-red-800 bg-red-900/20 text-red-400 text-sm">{error}</div>}

      {results && (
        <div className="space-y-4">
          {/* Best result highlight */}
          <div className="card border border-green-700 bg-green-900/10">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-green-400 font-bold text-lg">🏆 Best Parameters Found</div>
                <div className="text-white font-mono text-sm mt-1">
                  {Object.entries(results.best_params).map(([k, v]) => (
                    <span key={k} className="mr-3">
                      <span className="text-gray-400">{k}=</span>
                      <span className="text-green-300">{v}</span>
                    </span>
                  ))}
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-gray-400">{metricLabel[results.optimize_for]}</div>
                <div className="text-2xl font-bold text-green-400">{fmt.num(results.best_score)}</div>
                {results.improvement_pct !== 0 && (
                  <div className={`text-sm ${results.improvement_pct > 0 ? "text-green-400" : "text-red-400"}`}>
                    {results.improvement_pct > 0 ? "+" : ""}{results.improvement_pct.toFixed(1)}% vs default
                  </div>
                )}
              </div>
            </div>
            <div className="mt-3 text-xs text-gray-500">
              Tested {results.total_combinations_tested} combinations in {(results.processing_time_ms / 1000).toFixed(1)}s
              {" | "}Default score: {fmt.num(results.default_score)}
            </div>
          </div>

          {/* All results table */}
          <div className="card">
            <h3 className="text-sm font-semibold text-gray-400 mb-3 uppercase tracking-wider">
              All {results.total_combinations_tested} Combinations
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-xs text-gray-500 border-b border-gray-800">
                    <th className="text-left py-2">Parameters</th>
                    <th className="text-right py-2">Sharpe</th>
                    <th className="text-right py-2">Return</th>
                    <th className="text-right py-2">MDD</th>
                    <th className="text-right py-2">Win%</th>
                    <th className="text-right py-2">Trades</th>
                    <th className="text-right py-2">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {results.all_results.map((r, i) => {
                    const isBest = JSON.stringify(r.params) === JSON.stringify(results.best_params);
                    return (
                      <tr
                        key={i}
                        className={`border-b border-gray-800/50 ${isBest ? "bg-green-900/10" : "hover:bg-gray-800/30"}`}
                      >
                        <td className="py-2 font-mono text-xs text-gray-300">
                          {Object.entries(r.params).map(([k, v]) => `${k}=${v}`).join(", ")}
                          {isBest && <span className="ml-2 text-green-400">★</span>}
                        </td>
                        <td className={`text-right py-2 ${r.sharpe_ratio >= 1 ? "text-green-400" : r.sharpe_ratio >= 0.5 ? "text-yellow-400" : "text-red-400"}`}>
                          {fmt.num(r.sharpe_ratio)}
                        </td>
                        <td className={`text-right py-2 ${r.total_return >= 0 ? "text-green-400" : "text-red-400"}`}>
                          {fmt.pct(r.total_return)}
                        </td>
                        <td className={`text-right py-2 ${r.max_drawdown <= 0.15 ? "text-green-400" : r.max_drawdown <= 0.25 ? "text-yellow-400" : "text-red-400"}`}>
                          {fmt.pct(r.max_drawdown)}
                        </td>
                        <td className="text-right py-2 text-gray-300">{(r.win_rate * 100).toFixed(0)}%</td>
                        <td className="text-right py-2 text-gray-400">{r.n_trades}</td>
                        <td className="text-right py-2 text-white font-semibold">{fmt.num(r.score)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StrategyOptimizer;
