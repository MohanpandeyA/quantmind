/**
 * EarningsCalendar — upcoming earnings dates with risk warnings.
 *
 * WHY THIS COMPONENT EXISTS:
 *   Earnings announcements are the #1 driver of short-term price moves.
 *   A stock can move ±20% on earnings day. If you run QuantMind analysis
 *   2 days before earnings, the recommendation could be completely wrong
 *   after the announcement.
 *
 *   This component:
 *   1. Shows upcoming earnings for popular tickers
 *   2. Warns when earnings are within 7 days
 *   3. Lets you check any specific ticker
 *   4. Shows EPS and revenue estimates when available
 *
 * WARNING LEVELS:
 *   🚨 today    — Earnings TODAY, do not trade
 *   ⚠️ critical — Earnings in 1-3 days, HIGH RISK
 *   📅 warning  — Earnings in 4-7 days, be cautious
 *   ℹ️ info     — Earnings 8+ days away, safe to analyze
 */

import { useState, useEffect } from "react";
import axios from "axios";
import TickerAutocomplete from "./TickerAutocomplete";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

const WARNING_STYLES = {
  today: "border-red-700 bg-red-900/20 text-red-400",
  critical: "border-orange-700 bg-orange-900/20 text-orange-400",
  warning: "border-yellow-700 bg-yellow-900/20 text-yellow-400",
  info: "border-gray-700 bg-gray-800/30 text-gray-400",
  unknown: "border-gray-800 bg-gray-900/20 text-gray-500",
  past: "border-gray-800 bg-gray-900/10 text-gray-600",
  error: "border-gray-800 bg-gray-900/10 text-gray-600",
};

const EarningsCalendar = () => {
  const [calendar, setCalendar] = useState(null);
  const [loading, setLoading] = useState(false);
  const [singleTicker, setSingleTicker] = useState("");
  const [singleResult, setSingleResult] = useState(null);
  const [singleLoading, setSingleLoading] = useState(false);
  const [customTickers, setCustomTickers] = useState("AAPL,MSFT,GOOGL,NVDA,TSLA,JPM,AMZN,META");

  const fetchCalendar = async () => {
    setLoading(true);
    try {
      const resp = await axios.get(`${BASE_URL}/earnings/calendar/upcoming?tickers=${customTickers}`);
      setCalendar(resp.data);
    } catch (err) {
      console.error("Earnings calendar error:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCalendar();
  }, []);

  const checkSingleTicker = async (e) => {
    e.preventDefault();
    if (!singleTicker) return;
    setSingleLoading(true);
    setSingleResult(null);
    try {
      const resp = await axios.get(`${BASE_URL}/earnings/${singleTicker.toUpperCase()}`);
      setSingleResult(resp.data);
    } catch (err) {
      setSingleResult({ error: err.message });
    } finally {
      setSingleLoading(false);
    }
  };

  const fmt = {
    usd: (v) => v != null ? `$${(v / 1e9).toFixed(1)}B` : "—",
    eps: (v) => v != null ? `$${v.toFixed(2)}` : "—",
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white flex items-center gap-2">
        <span>📅</span> Earnings Calendar
      </h2>

      <div className="card bg-blue-900/10 border border-blue-800/50">
        <p className="text-sm text-blue-300">
          <strong>Why this matters:</strong> Stocks can move ±20% on earnings day.
          Always check upcoming earnings before acting on an analysis.
          Earnings within 7 days = HIGH RISK period.
        </p>
      </div>

      {/* Check single ticker */}
      <div className="card">
        <h3 className="text-sm font-semibold text-gray-400 mb-3">Check Specific Ticker</h3>
        <form onSubmit={checkSingleTicker} className="flex gap-3">
          <TickerAutocomplete
            value={singleTicker}
            onChange={(val) => setSingleTicker(val.toUpperCase())}
            onSelect={({ symbol }) => setSingleTicker(symbol)}
            placeholder="AAPL, RELIANCE.NS..."
            showHint={false}
            className="flex-1"
          />
          <button type="submit" disabled={singleLoading || !singleTicker} className="btn-primary px-6">
            {singleLoading ? "..." : "Check"}
          </button>
        </form>

        {singleResult && !singleResult.error && (
          <div className={`mt-3 p-3 rounded-lg border ${WARNING_STYLES[singleResult.warning_level]}`}>
            <div className="font-semibold">{singleResult.emoji} {singleResult.message}</div>
            {(singleResult.eps_estimate || singleResult.revenue_estimate) && (
              <div className="mt-2 text-xs flex gap-4">
                {singleResult.eps_estimate && (
                  <span>EPS Estimate: <strong>{fmt.eps(singleResult.eps_estimate)}</strong></span>
                )}
                {singleResult.revenue_estimate && (
                  <span>Revenue Estimate: <strong>{fmt.usd(singleResult.revenue_estimate)}</strong></span>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Calendar for multiple tickers */}
      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-400">Upcoming Earnings</h3>
          <button onClick={fetchCalendar} disabled={loading} className="text-xs text-blue-400 hover:text-blue-300">
            {loading ? "Loading..." : "🔄 Refresh"}
          </button>
        </div>

        <div className="flex gap-2 mb-3">
          <input
            value={customTickers}
            onChange={(e) => setCustomTickers(e.target.value)}
            placeholder="AAPL,MSFT,GOOGL..."
            className="input-field text-sm flex-1"
          />
          <button onClick={fetchCalendar} className="btn-primary text-sm px-4">Update</button>
        </div>

        {loading && (
          <div className="text-center text-gray-500 py-8">Loading earnings calendar...</div>
        )}

        {calendar && !loading && (
          <>
            {/* Urgent warnings */}
            {calendar.upcoming_within_7_days?.length > 0 && (
              <div className="mb-4 p-3 rounded-lg border border-orange-700 bg-orange-900/10">
                <div className="text-orange-400 font-semibold text-sm mb-2">
                  ⚠️ Earnings within 7 days — Exercise caution:
                </div>
                <div className="flex flex-wrap gap-2">
                  {calendar.upcoming_within_7_days.map((e) => (
                    <span key={e.ticker} className="text-xs bg-orange-900/30 text-orange-300 px-2 py-1 rounded-full">
                      {e.ticker} ({e.days_until_earnings}d)
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Full calendar table */}
            <div className="space-y-2">
              {calendar.calendar.map((item) => (
                <div
                  key={item.ticker}
                  className={`flex items-center justify-between p-3 rounded-lg border ${WARNING_STYLES[item.warning_level]}`}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-xl">{item.emoji}</span>
                    <div>
                      <div className="font-bold font-mono text-white">{item.ticker}</div>
                      <div className="text-xs opacity-80">{item.message}</div>
                    </div>
                  </div>
                  <div className="text-right text-xs">
                    {item.next_earnings_date && (
                      <div className="font-semibold">{item.next_earnings_date}</div>
                    )}
                    {item.eps_estimate && (
                      <div className="opacity-70">EPS est: {fmt.eps(item.eps_estimate)}</div>
                    )}
                    {item.days_until_earnings !== null && item.days_until_earnings >= 0 && (
                      <div className={`font-bold ${
                        item.days_until_earnings <= 3 ? "text-red-400" :
                        item.days_until_earnings <= 7 ? "text-yellow-400" : "text-gray-400"
                      }`}>
                        {item.days_until_earnings === 0 ? "TODAY" : `${item.days_until_earnings}d`}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {calendar.calendar.length === 0 && (
                <div className="text-center text-gray-500 py-6">
                  No upcoming earnings found for these tickers.
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default EarningsCalendar;
