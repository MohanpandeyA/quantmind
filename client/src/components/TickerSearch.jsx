import { useState } from "react";
import TickerAutocomplete from "./TickerAutocomplete";

const QUICK_PICKS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "JPM", "AMZN", "META"];

const TickerSearch = ({ onAnalyze, loading }) => {
  const [ticker, setTicker] = useState("");
  const [query, setQuery] = useState("Should I buy this stock?");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!ticker.trim() || loading) return;
    onAnalyze({ ticker: ticker.trim().toUpperCase(), query });
  };

  return (
    <div className="card">
      <p className="section-title mb-4">Analyze a Stock</p>

      <form onSubmit={handleSubmit} className="space-y-3">
        <div>
          <label className="text-xs font-medium text-slate-500 mb-1.5 block">Ticker Symbol</label>
          <TickerAutocomplete
            value={ticker}
            onChange={(val) => setTicker(val.toUpperCase())}
            onSelect={({ symbol }) => setTicker(symbol)}
            placeholder="AAPL, RELIANCE.NS..."
            showHint={true}
          />
        </div>

        <div>
          <label className="text-xs font-medium text-slate-500 mb-1.5 block">Your Question</label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Should I buy this stock?"
            rows={2}
            className="input-field resize-none"
          />
        </div>

        <button
          type="submit"
          disabled={!ticker.trim() || loading}
          className="btn-primary w-full"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
              Analyzing...
            </span>
          ) : (
            "Run Analysis"
          )}
        </button>
      </form>

      <div className="mt-4">
        <p className="text-xs text-slate-400 mb-2">Quick picks</p>
        <div className="flex flex-wrap gap-1.5">
          {QUICK_PICKS.map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTicker(t)}
              className={`chip text-xs font-mono ${ticker === t ? "chip-active" : "chip-inactive"}`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default TickerSearch;
