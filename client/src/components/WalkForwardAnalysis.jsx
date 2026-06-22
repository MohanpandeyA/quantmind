import { useState } from "react";
import axios from "axios";
import TickerAutocomplete from "./TickerAutocomplete";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

const VERDICT_CONFIG = {
  ROBUST: { color: "text-emerald-600", bg: "bg-emerald-50", border: "border-emerald-200", label: "Robust" },
  MODERATE: { color: "text-amber-600", bg: "bg-amber-50", border: "border-amber-200", label: "Moderate" },
  OVERFITTED: { color: "text-red-600", bg: "bg-red-50", border: "border-red-200", label: "Overfitted" },
};

const WalkForwardAnalysis = () => {
  const [ticker, setTicker] = useState("AAPL");
  const [strategy, setStrategy] = useState("macd");
  const [trainPeriod, setTrainPeriod] = useState("12mo");
  const [testPeriod, setTestPeriod] = useState("3mo");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runValidation = async () => {
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const resp = await axios.post(`${BASE_URL}/walk-forward`, {
        ticker, strategy, train_period: trainPeriod, test_period: testPeriod,
      });
      setResults(resp.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const verdict = results ? VERDICT_CONFIG[results.verdict] || VERDICT_CONFIG.MODERATE : null;

  const chartData = results?.windows?.map((w, i) => ({
    name: `W${i + 1}`,
    "In-Sample": parseFloat(w.train_sharpe?.toFixed(2)),
    "Out-of-Sample": parseFloat(w.test_sharpe?.toFixed(2)),
  })) || [];

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-slate-900">Walk-Forward Validation</h2>

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
            <label className="text-xs font-medium text-slate-500 mb-1.5 block">Train Period</label>
            <select value={trainPeriod} onChange={(e) => setTrainPeriod(e.target.value)} className="input-field">
              <option value="6mo">6 months</option>
              <option value="12mo">12 months</option>
              <option value="18mo">18 months</option>
            </select>
          </div>
          <div>
            <label className="text-xs font-medium text-slate-500 mb-1.5 block">Test Period</label>
            <select value={testPeriod} onChange={(e) => setTestPeriod(e.target.value)} className="input-field">
              <option value="3mo">3 months</option>
              <option value="6mo">6 months</option>
            </select>
          </div>
        </div>
        <button onClick={runValidation} disabled={loading} className="btn-primary w-full mt-4">
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
              Running validation...
            </span>
          ) : "Run Walk-Forward Test"}
        </button>
      </div>

      {error && <div className="card border-red-100 bg-red-50"><p className="text-red-600 text-sm">{error}</p></div>}

      {results && verdict && (
        <>
          <div className={`card border ${verdict.border} ${verdict.bg}`}>
            <div className="flex items-center justify-between">
              <div>
                <p className="section-title mb-1">Verdict</p>
                <div className={`text-2xl font-bold ${verdict.color}`}>{verdict.label}</div>
                <p className="text-xs text-slate-500 mt-1">
                  Robustness ratio: {results.robustness_ratio?.toFixed(2)} (target: &gt;0.7)
                </p>
              </div>
              <div className="text-right space-y-1">
                <div>
                  <div className="text-xs text-slate-400">In-Sample Sharpe</div>
                  <div className="text-lg font-bold text-slate-700">{results.in_sample_sharpe?.toFixed(2)}</div>
                </div>
                <div>
                  <div className="text-xs text-slate-400">Out-of-Sample Sharpe</div>
                  <div className={`text-lg font-bold ${verdict.color}`}>{results.out_of_sample_sharpe?.toFixed(2)}</div>
                </div>
              </div>
            </div>
          </div>

          {chartData.length > 0 && (
            <div className="card">
              <p className="section-title mb-4">Window Results</p>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={chartData} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: "white", border: "1px solid #e2e8f0", borderRadius: "12px" }} />
                  <ReferenceLine y={0} stroke="#e2e8f0" />
                  <Bar dataKey="In-Sample" fill="#c7d2fe" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Out-of-Sample" fill="#4f46e5" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default WalkForwardAnalysis;
