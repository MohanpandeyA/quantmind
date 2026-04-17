/**
 * TickerAutocomplete — reusable ticker input with company name autocomplete.
 *
 * Extracted from TickerSearch.jsx so it can be used in any form:
 *   - PortfolioTracker (Add Position)
 *   - StrategyOptimizer
 *   - PriceAlerts (Add Alert)
 *   - EarningsCalendar (Check Ticker)
 *
 * Props:
 *   value        {string}   — controlled input value (ticker symbol)
 *   onChange     {function} — called with new ticker string on every keystroke
 *   onSelect     {function} — called with {symbol, name} when user picks from dropdown
 *   placeholder  {string}   — input placeholder text
 *   disabled     {boolean}  — disable input (e.g. while loading)
 *   className    {string}   — extra CSS classes for the wrapper div
 *   showHint     {boolean}  — show Indian stock format hint (default true)
 *
 * Usage:
 *   <TickerAutocomplete
 *     value={ticker}
 *     onChange={(val) => setTicker(val)}
 *     onSelect={({ symbol }) => setTicker(symbol)}
 *     placeholder="AAPL, Reliance, Tesla..."
 *   />
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { searchTicker } from "../services/api";

const TickerAutocomplete = ({
  value,
  onChange,
  onSelect,
  placeholder = "e.g. Apple, Reliance, AAPL, RELIANCE.NS...",
  disabled = false,
  className = "",
  showHint = true,
}) => {
  const [suggestions, setSuggestions] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedName, setSelectedName] = useState("");
  const debounceRef = useRef(null);
  const dropdownRef = useRef(null);
  const inputRef = useRef(null);

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
  const handleInput = useCallback(
    (inputValue) => {
      onChange(inputValue);
      setSelectedName(""); // clear company name when user edits

      if (debounceRef.current) clearTimeout(debounceRef.current);

      if (inputValue.trim().length < 2) {
        setSuggestions([]);
        setShowDropdown(false);
        return;
      }

      debounceRef.current = setTimeout(async () => {
        setSearchLoading(true);
        try {
          const data = await searchTicker(inputValue.trim());
          setSuggestions(data.results || []);
          setShowDropdown((data.results || []).length > 0);
        } catch {
          setSuggestions([]);
          setShowDropdown(false);
        } finally {
          setSearchLoading(false);
        }
      }, 350);
    },
    [onChange]
  );

  // User selects a suggestion
  const handleSelect = (result) => {
    onChange(result.symbol);
    setSelectedName(result.name);
    setSuggestions([]);
    setShowDropdown(false);
    if (onSelect) onSelect(result);
  };

  // Clear button
  const handleClear = () => {
    onChange("");
    setSelectedName("");
    setSuggestions([]);
    setShowDropdown(false);
    inputRef.current?.focus();
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
    <div className={`relative ${className}`} ref={dropdownRef}>
      {/* Input */}
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => handleInput(e.target.value)}
          onFocus={() => suggestions.length > 0 && setShowDropdown(true)}
          placeholder={placeholder}
          className="input-field pr-8 w-full"
          disabled={disabled}
          autoComplete="off"
        />
        {/* Spinner or clear button */}
        <div className="absolute right-2 top-1/2 -translate-y-1/2">
          {searchLoading ? (
            <svg className="animate-spin h-4 w-4 text-gray-500" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          ) : value ? (
            <button
              type="button"
              onClick={handleClear}
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
                <span className="text-sm font-mono font-bold text-white bg-gray-700 group-hover:bg-gray-600 px-2 py-0.5 rounded shrink-0">
                  {result.symbol}
                </span>
                <span className="text-sm text-gray-300 truncate">{result.name}</span>
              </div>
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

      {/* Format hint */}
      {showHint && (
        <p className="text-xs text-gray-600 mt-1">
          Indian stocks: add{" "}
          <span className="text-gray-500">.NS</span> (NSE) or{" "}
          <span className="text-gray-500">.BO</span> (BSE) — e.g.{" "}
          <span className="text-gray-500">RELIANCE.NS</span>
        </p>
      )}
    </div>
  );
};

export default TickerAutocomplete;
