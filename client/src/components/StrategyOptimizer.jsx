import { useState } from "react";
import axios from "axios";
import TickerAutocomplete from "./TickerAutocomplete";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

const DEFAULT_GRIDS = {
  momentum: { short_window: [10, 20, 30], long_window: [40, 50, 60] },
  mean_reversion: { window: [10, 20, 30], z_threshold: [1.5, 2.0, 2.5] },
  rsi: { period: [7, 14, 21], oversold: [25, 30, 35], overbought: [65, 70, 75] },
  macd: { fast: [8, 12, 16], slow: [21, 26, 30], signal_period: [7, 9, 12] },
};

const StrategyOptimizer = () => {
  const [ticker, setTicker] = useState("AAPL");
  const [strategy, setStrategy] = useState("macd");
  const [optimizeFor, setOptimizeFor] = useState("sharpe");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runOptimize = async () => {
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const resp = await axios.post(`${BASE_URL}/optimize`, {
        ticker, strategy, optimize_for: optimizeFor,
        param_grid: DEFAULT_GRIDS[strategy],
      });
      setResults(resp.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const chartData = results?.all_results
    ?.sort((a, b) => (b.sharpe_ratio || 0) - (a.sharpe_ratio || 0))
    ?.slice(0, 10)
    ?.map((r, i) => ({
      name: `#${i + 1}`,
      Sharpe: parseFloat((r.sharpe_ratio || 0).toFixed(2)),
      Return: parseFloat(((r.total_return || 0) * 100).toFixed(1)),
    })) || [];

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-slate-900">Strategy Optimizer</h2>

      <div className="card">
        <p className="section-title mb-4">Configuration</p>
        <div className="grid grid-cols-2 gap-3">
          <div className="col-span-2 sm:col-span-1">
            <label className="text-xs font-medium text-slate-500 mb-1.5 block">Ticker</label>
            <TickerAutocomplete value={ticker} onChange={(v) => setTicker(v.toUpperCase())} onSelect={({ symbol }) => setTicker(symbol)} placeholder="AAPL" showHint={false} />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-500 mb-1.5 block">Strategy</label>
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)} className="input-field">
              <option value="momentum">Momentum</option>
              <option value="mean_reversion">Mean Reversion</option>
              <option value="rsi">RSI</option>
              <option value="macd">MACD</option>
            </select>
          </div>
          <div>
            <label className="text-xs font-medium text-slate-500 mb-1.5 block">Optimize For</label>
            <select value={optimizeFor} onChange={(e) => setOptimizeFor(e.target.value)} className="input-field">
              <option value="sharpe">Sharpe Ratio</option>
              <option value="return">Total Return</option>
              <option value="calmar">Calmar Ratio</option>
            </select>
          </div>
        </div>
        <button onClick={runOptimize} disabled={loading} className="btn-primary w-full mt-4">
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
              Running grid search...
            </span>
          ) : "Run Optimization"}
        </button>
      </div>

      {error && <div className="card border-red-100 bg-red-50"><p className="text-red-600 text-sm">{error}</p></div>}

      {results && (
        <>
          <div className="card border-indigo-100 bg-indigo-50">
            <p className="section-title mb-2">Best Parameters</p>
            <div className="flex flex-wrap gap-2 mb-3">
              {Object.entries(results.best_params || {}).map(([k, v]) => (
                <div key={k} className="bg-white border border-indigo-100 rounded-lg px-3 py-1.5">
                  <span className="text-xs text-slate-400">{k}: </span>
                  <span className="text-sm font-semibold text-indigo-700">{v}</span>
                </div>
              ))}
            </div>
            <div className="flex gap-4 text-sm">
              <div>
                <span className="text-slate-400 text-xs">Best Score</span>
                <div className="font-bold text-indigo-700">{results.best_score?.toFixed(3)}</div>
              </div>
              {results.improvement_pct > 0 && (
                <div>
                  <span className="text-slate-400 text-xs">Improvement</span>
                  <div className="font-bold text-emerald-600">+{results.improvement_pct?.toFixed(1)}%</div>
                </div>
              )}
            </div>
          </div>

          {chartData.length > 0 && (
            <div className="card">
              <p className="section-title mb-4">Top 10 Parameter Combinations</p>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={chartData} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: "white", border: "1px solid #e2e8f0", borderRadius: "12px" }} />
                  <Bar dataKey="Sharpe" fill="#4f46e5" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default StrategyOptimizer;
