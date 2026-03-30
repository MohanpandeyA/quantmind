/**
 * TickerSearch — search form for triggering trading analysis.
 *
 * Allows the user to enter:
 * - Ticker symbol (AAPL, MSFT, RELIANCE.NS)
 * - Natural language query
 * - Optional date range for backtesting
 *
 * Calls useAnalysis hook to trigger the LangGraph workflow.
 */

import { useState } from "react";

/**
 * @param {Object} props
 * @param {Function} props.onAnalyze - Called with {ticker, query, startDate, endDate}
 * @param {boolean} props.loading - True while analysis is running
 */
const TickerSearch = ({ onAnalyze, loading }) => {
  const [ticker, setTicker] = useState("");
  const [query, setQuery] = useState("");
  const [startDate, setStartDate] = useState("2022-01-01");
  const [endDate, setEndDate] = useState("2024-12-31");
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!ticker.trim() || !query.trim()) return;
    onAnalyze({ ticker: ticker.trim(), query: query.trim(), startDate, endDate });
  };

  // Quick query suggestions
  const suggestions = [
    "Should I buy this stock?",
    "What are the key risk factors?",
    "How has revenue trended recently?",
    "Is this a good entry point?",
  ];

  return (
    <div className="card">
      <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
        <span className="text-blue-400">🔍</span>
        Analyze a Stock
      </h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Ticker input */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Ticker Symbol
          </label>
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="AAPL, MSFT, GOOGL, RELIANCE.NS..."
            className="input-field"
            maxLength={20}
            disabled={loading}
          />
        </div>

        {/* Query input */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Your Question
          </label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Should I buy this stock given current market conditions?"
            className="input-field resize-none"
            rows={3}
            maxLength={500}
            disabled={loading}
          />
          {/* Quick suggestions */}
          <div className="flex flex-wrap gap-2 mt-2">
            {suggestions.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setQuery(s)}
                className="text-xs text-blue-400 hover:text-blue-300 bg-blue-900/30
                           hover:bg-blue-900/50 px-2 py-1 rounded-md transition-colors"
                disabled={loading}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        {/* Advanced: date range */}
        <div>
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-sm text-gray-400 hover:text-gray-300 flex items-center gap-1"
          >
            <span>{showAdvanced ? "▼" : "▶"}</span>
            Advanced (backtest date range)
          </button>

          {showAdvanced && (
            <div className="grid grid-cols-2 gap-4 mt-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Start Date</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="input-field text-sm"
                  disabled={loading}
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">End Date</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="input-field text-sm"
                  disabled={loading}
                />
              </div>
            </div>
          )}
        </div>

        {/* Submit button */}
        <button
          type="submit"
          disabled={loading || !ticker.trim() || !query.trim()}
          className="btn-primary w-full flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Analyzing {ticker}...
            </>
          ) : (
            <>
              <span>⚡</span>
              Run Analysis
            </>
          )}
        </button>
      </form>

      {loading && (
        <div className="mt-4 text-sm text-gray-400 text-center space-y-1">
          <p>🔍 Fetching market data...</p>
          <p>📚 Retrieving SEC filings & news...</p>
          <p>📊 Running backtest...</p>
          <p>🤖 Generating AI explanation...</p>
        </div>
      )}
    </div>
  );
};

export default TickerSearch;
