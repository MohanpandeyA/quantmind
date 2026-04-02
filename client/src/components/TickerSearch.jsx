/**
 * TickerSearch — search form with company name autocomplete.
 *
 * Users can type a company name (e.g., "reliance", "apple") or a partial
 * ticker symbol and get a dropdown of matching tickers with company names.
 * Selecting a result fills the ticker field with the correct symbol.
 *
 * Autocomplete flow:
 *   user types "reliance"
 *   → debounced 350ms
 *   → GET /api/ticker/search?q=reliance (FastAPI → Yahoo Finance)
 *   → dropdown shows: RELIANCE.NS — Reliance Industries Limited (NSI)
 *   → user clicks → ticker field = "RELIANCE.NS"
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { searchTicker } from "../services/api";

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

  // Autocomplete state
  const [suggestions, setSuggestions] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedName, setSelectedName] = useState(""); // company name display
  const debounceRef = useRef(null);
  const dropdownRef = useRef(null);
  const inputRef = useRef(null);

  // Quick query suggestions
  const querySuggestions = [
    "Should I buy this stock?",
    "What are the key risk factors?",
    "How has revenue trended recently?",
    "Is this a good entry point?",
  ];

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Debounced search — fires 350ms after user stops typing
  const handleTickerInput = useCallback((value) => {
    setTicker(value);
    setSelectedName(""); // clear company name when user edits

    // Clear previous debounce
    if (debounceRef.current) clearTimeout(debounceRef.current);

    // Don't search if too short
    if (value.trim().length < 2) {
      setSuggestions([]);
      setShowDropdown(false);
      return;
    }

    debounceRef.current = setTimeout(async () => {
      setSearchLoading(true);
      try {
        const data = await searchTicker(value.trim());
        setSuggestions(data.results || []);
        setShowDropdown((data.results || []).length > 0);
      } catch {
        setSuggestions([]);
        setShowDropdown(false);
      } finally {
        setSearchLoading(false);
      }
    }, 350);
  }, []);

  // User selects a suggestion from dropdown
  const handleSelect = (result) => {
    setTicker(result.symbol);
    setSelectedName(result.name);
    setSuggestions([]);
    setShowDropdown(false);
    // Focus query field after selection
    setTimeout(() => {
      const queryEl = document.getElementById("qm-query-input");
      if (queryEl) queryEl.focus();
    }, 50);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!ticker.trim() || !query.trim()) return;
    setShowDropdown(false);
    onAnalyze({ ticker: ticker.trim(), query: query.trim(), startDate, endDate });
  };

  // Exchange badge color
  const exchangeColor = (exchange) => {
    if (!exchange) return "text-gray-500";
    const e = exchange.toUpperCase();
    if (e.includes("NS") || e.includes("NSI") || e.includes("BSE")) return "text-orange-400";
    if (e.includes("NAS")) return "text-blue-400";
    if (e.includes("NYS")) return "text-green-400";
    return "text-gray-400";
  };

  return (
    <div className="card">
      <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
        <span className="text-blue-400">🔍</span>
        Analyze a Stock
      </h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Ticker input with autocomplete */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Ticker Symbol
            <span className="text-gray-600 font-normal ml-2">— type a company name or symbol</span>
          </label>

          <div className="relative" ref={dropdownRef}>
            <div className="relative">
              <input
                ref={inputRef}
                type="text"
                value={ticker}
                onChange={(e) => handleTickerInput(e.target.value)}
                onFocus={() => suggestions.length > 0 && setShowDropdown(true)}
                placeholder="e.g. Apple, Reliance, AAPL, RELIANCE.NS..."
                className="input-field pr-8"
                maxLength={20}
                disabled={loading}
                autoComplete="off"
              />
              {/* Search spinner or clear button */}
              <div className="absolute right-2 top-1/2 -translate-y-1/2">
                {searchLoading ? (
                  <svg className="animate-spin h-4 w-4 text-gray-500" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                ) : ticker ? (
                  <button
                    type="button"
                    onClick={() => { setTicker(""); setSelectedName(""); setSuggestions([]); setShowDropdown(false); inputRef.current?.focus(); }}
                    className="text-gray-500 hover:text-gray-300 text-lg leading-none"
                  >
                    ×
                  </button>
                ) : null}
              </div>
            </div>

            {/* Company name display after selection */}
            {selectedName && !showDropdown && (
              <p className="text-xs text-green-400 mt-1 flex items-center gap-1">
                <span>✓</span> {selectedName}
              </p>
            )}

            {/* Autocomplete dropdown */}
            {showDropdown && suggestions.length > 0 && (
              <div className="absolute z-50 w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-xl overflow-hidden">
                {suggestions.map((result) => (
                  <button
                    key={result.symbol}
                    type="button"
                    onClick={() => handleSelect(result)}
                    className="w-full px-3 py-2.5 flex items-center justify-between hover:bg-gray-700 transition-colors text-left group"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      {/* Symbol badge */}
                      <span className="text-sm font-mono font-bold text-white bg-gray-700 group-hover:bg-gray-600 px-2 py-0.5 rounded shrink-0">
                        {result.symbol}
                      </span>
                      {/* Company name */}
                      <span className="text-sm text-gray-300 truncate">{result.name}</span>
                    </div>
                    {/* Exchange + type */}
                    <div className="flex items-center gap-2 shrink-0 ml-2">
                      <span className={`text-xs font-medium ${exchangeColor(result.exchange)}`}>
                        {result.exchange}
                      </span>
                      <span className="text-xs text-gray-600">{result.type_display}</span>
                    </div>
                  </button>
                ))}
                <div className="px-3 py-1.5 border-t border-gray-700 text-xs text-gray-600">
                  Powered by Yahoo Finance
                </div>
              </div>
            )}
          </div>

          {/* Format hint */}
          <p className="text-xs text-gray-600 mt-1">
            Indian stocks: add <span className="text-gray-500">.NS</span> (NSE) or <span className="text-gray-500">.BO</span> (BSE) — e.g. <span className="text-gray-500">RELIANCE.NS</span>
          </p>
        </div>

        {/* Query input */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Your Question
          </label>
          <textarea
            id="qm-query-input"
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
            {querySuggestions.map((s) => (
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
