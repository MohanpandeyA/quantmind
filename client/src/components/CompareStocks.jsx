import { useState } from "react";
import axios from "axios";
import TickerAutocomplete from "./TickerAutocomplete";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

const SIGNAL_COLORS = {
  BUY: "text-emerald-600 bg-emerald-50 border-emerald-100",
  SELL: "text-red-600 bg-red-50 border-red-100",
  HOLD: "text-amber-600 bg-amber-50 border-amber-100",
};

const CompareStocks = () => {
  const [tickers, setTickers] = useState(["AAPL", "MSFT"]);
  const [query, setQuery] = useState("Which stock should I buy?");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [newTicker, setNewTicker] = useState("");

  const addTicker = () => {
    if (newTicker && !tickers.includes(newTicker.toUpperCase()) && tickers.length < 5) {
      setTickers([...tickers, newTicker.toUpperCase()]);
      setNewTicker("");
    }
  };

  const removeTicker = (t) => setTickers(tickers.filter((x) => x !== t));

  const runCompare = async () => {
    if (tickers.length < 2) return;
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const resp = await axios.post(`${BASE_URL}/compare`, { tickers, query });
      setResults(resp.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-slate-900">Compare Stocks</h2>

      <div className="card">
        <p className="section-title mb-4">Select Tickers (2-5)</p>

        <div className="flex flex-wrap gap-2 mb-3">
          {tickers.map((t) => (
            <div key={t} className="flex items-center gap-1.5 bg-indigo-50 border border-indigo-100 text-indigo-700 px-3 py-1.5 rounded-lg text-sm font-mono font-medium">
              {t}
              <button onClick={() => removeTicker(t)} className="text-indigo-400 hover:text-indigo-700 ml-1 text-xs">x</button>
            </div>
          ))}
        </div>

        <div className="flex gap-2 mb-4">
          <TickerAutocomplete
            value={newTicker}
            onChange={(v) => setNewTicker(v.toUpperCase())}
            onSelect={({ symbol }) => { setNewTicker(symbol); }}
            placeholder="Add ticker..."
            showHint={false}
            className="flex-1"
          />
          <button onClick={addTicker} disabled={!newTicker || tickers.length >= 5} className="btn-secondary px-4">
            Add
          </button>
        </div>

        <div className="mb-4">
          <label className="text-xs font-medium text-slate-500 mb-1.5 block">Question</label>
          <input value={query} onChange={(e) => setQuery(e.target.value)} className="input-field" />
        </div>

        <button onClick={runCompare} disabled={tickers.length < 2 || loading} className="btn-primary w-full">
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
              Comparing {tickers.length} stocks...
            </span>
          ) : `Compare ${tickers.length} Stocks`}
        </button>
      </div>

      {error && <div className="card border-red-100 bg-red-50"><p className="text-red-600 text-sm">{error}</p></div>}

      {results && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="section-title">Rankings</p>
            <span className="badge badge-indigo">Best: {results.best_ticker}</span>
          </div>

          {results.rankings?.map((r) => (
            <div key={r.ticker} className={`card ${r.rank === 1 ? "border-indigo-200 bg-indigo-50/30" : ""}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold ${r.rank === 1 ? "bg-indigo-600 text-white" : "bg-slate-100 text-slate-500"}`}>
                    {r.rank}
                  </div>
                  <div>
                    <div className="font-bold text-slate-900 font-mono">{r.ticker}</div>
                    <div className="text-xs text-slate-400 capitalize">{r.strategy?.replace(/_/g, " ")}</div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-right hidden sm:block">
                    <div className="text-sm font-semibold text-slate-700">Score: {r.composite_score?.toFixed(3)}</div>
                    <div className="text-xs text-slate-400">Sharpe: {r.sharpe_ratio?.toFixed(2)}</div>
                  </div>
                  <span className={`badge border ${SIGNAL_COLORS[r.signal] || SIGNAL_COLORS.HOLD}`}>{r.signal}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default CompareStocks;
