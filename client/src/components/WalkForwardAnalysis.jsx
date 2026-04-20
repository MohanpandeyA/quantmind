/**
 * WalkForwardAnalysis — tests if a strategy is robust (not overfitted).
 *
 * Walk-forward validation splits data into rolling train/test windows:
 *   Train: optimize parameters on historical data
 *   Test:  apply those parameters to UNSEEN future data
 *
 * If the strategy performs well on unseen data → ROBUST (not overfitted)
 * If it only works on historical data → OVERFITTED (avoid it)
 *
 * Robustness ratio = out-of-sample Sharpe / in-sample Sharpe
 *   > 0.7  → ROBUST
 *   0.4-0.7 → MODERATE
 *   < 0.4  → OVERFITTED
 */

import { useState } from "react";
import axios from "axios";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import TickerAutocomplete from "./TickerAutocomplete";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

const VERDICT_STYLES = {
  ROBUST: {
    bg: "bg-green-900/20 border-green-700",
    text: "text-green-400",
    icon: "✅",
    desc: "Strategy holds up on unseen data — low overfitting risk",
  },
  MODERATE: {
    bg: "bg-yellow-900/20 border-yellow-700",
    text: "text-yellow-400",
    icon: "⚠️",
    desc: "Some overfitting detected — use with caution",
  },
  OVERFITTED: {
    bg: "bg-red-900/20 border-red-700",
    text: "text-red-400",
    icon: "❌",
    desc: "Strategy only works on historical data — avoid in live trading",
  },
};

const WalkForwardAnalysis = () => {
  const [ticker, setTicker] = useState("AAPL");
  const [strategy, setStrategy] = useState("macd");
  const [startDate, setStartDate] = useState("2022-01-01");
  const [endDate, setEndDate] = useState("2024-12-31");
  const [trainMonths, setTrainMonths] = useState(12);
  const [testMonths, setTestMonths] = useState(3);
  const [optimizeFor, setOptimizeFor] = useState("sharpe");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runValidation = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const resp = await axios.post(`${BASE_URL}/walk-forward`, {
        ticker: ticker.toUpperCase(),
        strategy,
        start_date: startDate,
        end_date: endDate,
        train_months: parseInt(trainMonths),
        test_months: parseInt(testMonths),
        step_months: parseInt(testMonths), // step = test window size
        optimize_for: optimizeFor,
      });
      setResult(resp.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const fmt = {
    pct: (v) => v != null ? `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%` : "—",
    sharpe: (v) => v != null ? v.toFixed(3) : "—",
    ratio: (v) => v != null ? v.toFixed(2) : "—",
  };

  // Build equity curve chart data
  const equityData = result?.combined_equity_curve?.map((v, i) => ({
    day: i,
    value: Math.round(v),
  })) || [];

  const verdict = result ? VERDICT_STYLES[result.verdict] || VERDICT_STYLES.MODERATE : null;

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white flex items-center gap-2">
        <span>🔬</span> Walk-Forward Validation
      </h2>

      {/* Explainer */}
      <div className="card bg-blue-900/10 border border-blue-800/50">
        <p className="text-sm text-blue-300">
          <strong>What this does:</strong> Tests if a strategy is <em>robust</em> or <em>overfitted</em>.
          Splits data into rolling train/test windows — optimizes on train, tests on unseen data.
          If out-of-sample performance ≈ in-sample → strategy is trustworthy.
        </p>
      </div>

      {/* Form */}
      <div className="card">
        <form onSubmit={runValidation} className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Ticker</label>
              <TickerAutocomplete
                value={ticker}
                onChange={(val) => setTicker(val.toUpperCase())}
                onSelect={({ symbol }) => setTicker(symbol)}
                placeholder="AAPL, RELIANCE.NS..."
                disabled={loading}
                showHint={false}
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
                <option value="momentum">Momentum (EMA Crossover)</option>
                <option value="mean_reversion">Mean Reversion (Z-Score)</option>
                <option value="rsi">RSI (Overbought/Oversold)</option>
                <option value="macd">MACD (Triple EMA)</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Train Window</label>
              <select
                value={trainMonths}
                onChange={(e) => setTrainMonths(e.target.value)}
                className="input-field text-sm"
                disabled={loading}
              >
                <option value={6}>6 months</option>
                <option value={9}>9 months</option>
                <option value={12}>12 months</option>
                <option value={18}>18 months</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Test Window</label>
              <select
                value={testMonths}
                onChange={(e) => setTestMonths(e.target.value)}
                className="input-field text-sm"
                disabled={loading}
              >
                <option value={1}>1 month</option>
                <option value={2}>2 months</option>
                <option value={3}>3 months</option>
                <option value={6}>6 months</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Optimize For</label>
              <select
                value={optimizeFor}
                onChange={(e) => setOptimizeFor(e.target.value)}
                className="input-field text-sm"
                disabled={loading}
              >
                <option value="sharpe">Sharpe Ratio</option>
                <option value="total_return">Total Return</option>
                <option value="calmar">Calmar Ratio</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Start Date</label>
              <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)}
                className="input-field text-sm" disabled={loading} />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">End Date</label>
              <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)}
                className="input-field text-sm" disabled={loading} />
            </div>
          </div>

          <button type="submit" disabled={loading || !ticker.trim()} className="btn-primary w-full flex items-center justify-center gap-2">
            {loading ? (
              <><svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
              </svg> Running validation...</>
            ) : "🔬 Run Walk-Forward Validation"}
          </button>
        </form>
      </div>

      {error && (
        <div className="card border border-red-800 bg-red-900/20 text-red-400 text-sm">{error}</div>
      )}

      {result && (
        <>
          {/* Verdict banner */}
          <div className={`card border ${verdict.bg}`}>
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-3">
                  <span className="text-2xl">{verdict.icon}</span>
                  <div>
                    <div className={`text-xl font-bold ${verdict.text}`}>
                      {result.verdict}
                    </div>
                    <div className="text-sm text-gray-400">{verdict.desc}</div>
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-gray-500">Robustness Ratio</div>
                <div className={`text-3xl font-bold font-mono ${verdict.text}`}>
                  {fmt.ratio(result.robustness_ratio)}
                </div>
                <div className="text-xs text-gray-500">out-of-sample / in-sample</div>
              </div>
            </div>
          </div>

          {/* In-sample vs Out-of-sample comparison */}
          <div className="grid grid-cols-3 gap-4">
            <div className="card text-center">
              <div className="metric-label">In-Sample Sharpe</div>
              <div className="metric-value text-blue-400">{fmt.sharpe(result.in_sample_sharpe)}</div>
              <div className="text-xs text-gray-600 mt-1">Training data</div>
            </div>
            <div className="card text-center">
              <div className="metric-label">Out-of-Sample Sharpe</div>
              <div className={`metric-value ${result.out_of_sample_sharpe >= 0.5 ? "text-green-400" : result.out_of_sample_sharpe >= 0 ? "text-yellow-400" : "text-red-400"}`}>
                {fmt.sharpe(result.out_of_sample_sharpe)}
              </div>
              <div className="text-xs text-gray-600 mt-1">Unseen test data</div>
            </div>
            <div className="card text-center">
              <div className="metric-label">Windows Tested</div>
              <div className="metric-value text-white">{result.n_windows}</div>
              <div className="text-xs text-gray-600 mt-1">
                {(result.processing_time_ms / 1000).toFixed(1)}s total
              </div>
            </div>
          </div>

          {/* Out-of-sample equity curve */}
          {equityData.length > 1 && (
            <div className="card">
              <h3 className="text-sm font-semibold text-gray-400 mb-3">
                📈 Out-of-Sample Equity Curve (stitched test windows)
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={equityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="day" tick={{ fill: "#6B7280", fontSize: 10 }} tickLine={false} />
                  <YAxis tick={{ fill: "#6B7280", fontSize: 10 }} tickLine={false}
                    tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#1F2937", border: "1px solid #374151", borderRadius: "8px" }}
                    formatter={(v) => [`$${v.toLocaleString()}`, "Portfolio"]}
                    labelFormatter={(l) => `Day ${l}`}
                  />
                  <ReferenceLine y={100000} stroke="#6B7280" strokeDasharray="4 4" />
                  <Line type="monotone" dataKey="value" stroke="#60A5FA" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              <div className="flex justify-between text-xs text-gray-500 mt-2">
                <span>Out-of-sample return: <span className={result.combined_return >= 0 ? "text-green-400" : "text-red-400"}>{fmt.pct(result.combined_return)}</span></span>
                <span>Max drawdown: <span className="text-red-400">{fmt.pct(result.combined_max_drawdown)}</span></span>
              </div>
            </div>
          )}

          {/* Window-by-window results */}
          <div className="card">
            <h3 className="text-sm font-semibold text-gray-400 mb-3">
              📋 Window-by-Window Results
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-xs text-gray-500 border-b border-gray-800">
                    <th className="text-left py-2 pr-4">Window</th>
                    <th className="text-left py-2 pr-4">Train Period</th>
                    <th className="text-left py-2 pr-4">Test Period</th>
                    <th className="text-right py-2 pr-4">Best Params</th>
                    <th className="text-right py-2 pr-4">Train Sharpe</th>
                    <th className="text-right py-2 pr-4">Test Sharpe</th>
                    <th className="text-right py-2">Test Return</th>
                  </tr>
                </thead>
                <tbody>
                  {result.windows.map((w) => (
                    <tr key={w.window_idx} className="border-b border-gray-800/50 hover:bg-gray-800/20">
                      <td className="py-2 pr-4 text-gray-400">#{w.window_idx}</td>
                      <td className="py-2 pr-4 text-gray-500 text-xs">
                        {w.train_start.slice(0, 7)} → {w.train_end.slice(0, 7)}
                      </td>
                      <td className="py-2 pr-4 text-gray-500 text-xs">
                        {w.test_start.slice(0, 7)} → {w.test_end.slice(0, 7)}
                      </td>
                      <td className="py-2 pr-4 text-right text-xs text-blue-400 font-mono">
                        {Object.entries(w.best_params || {}).map(([k, v]) => `${k}=${v}`).join(", ")}
                      </td>
                      <td className="py-2 pr-4 text-right text-blue-400 font-mono">
                        {fmt.sharpe(w.train_sharpe)}
                      </td>
                      <td className={`py-2 pr-4 text-right font-mono ${w.test_sharpe >= 0.5 ? "text-green-400" : w.test_sharpe >= 0 ? "text-yellow-400" : "text-red-400"}`}>
                        {fmt.sharpe(w.test_sharpe)}
                      </td>
                      <td className={`py-2 text-right font-mono ${w.test_return >= 0 ? "text-green-400" : "text-red-400"}`}>
                        {fmt.pct(w.test_return)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Interpretation guide */}
          <div className="card bg-gray-800/30 border border-gray-700">
            <h3 className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wider">
              How to interpret
            </h3>
            <div className="grid grid-cols-3 gap-3 text-xs">
              <div className="flex items-start gap-2">
                <span className="text-green-400 shrink-0">✅</span>
                <div>
                  <div className="text-green-400 font-medium">ROBUST (ratio &gt; 0.7)</div>
                  <div className="text-gray-500">Strategy works on unseen data. Safe to use.</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-yellow-400 shrink-0">⚠️</span>
                <div>
                  <div className="text-yellow-400 font-medium">MODERATE (0.4–0.7)</div>
                  <div className="text-gray-500">Some overfitting. Reduce position size.</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-red-400 shrink-0">❌</span>
                <div>
                  <div className="text-red-400 font-medium">OVERFITTED (&lt; 0.4)</div>
                  <div className="text-gray-500">Only works on historical data. Avoid.</div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default WalkForwardAnalysis;
