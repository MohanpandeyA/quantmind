import { useState, useEffect } from "react";
import axios from "axios";
import TickerAutocomplete from "./TickerAutocomplete";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

const LEVEL_CONFIG = {
  today:    { bg: "bg-red-50",    border: "border-red-200",    text: "text-red-700",    badge: "bg-red-100 text-red-700" },
  critical: { bg: "bg-orange-50", border: "border-orange-200", text: "text-orange-700", badge: "bg-orange-100 text-orange-700" },
  warning:  { bg: "bg-amber-50",  border: "border-amber-200",  text: "text-amber-700",  badge: "bg-amber-100 text-amber-700" },
  info:     { bg: "bg-slate-50",  border: "border-slate-200",  text: "text-slate-600",  badge: "bg-slate-100 text-slate-600" },
  unknown:  { bg: "bg-slate-50",  border: "border-slate-100",  text: "text-slate-500",  badge: "bg-slate-100 text-slate-500" },
  past:     { bg: "bg-slate-50",  border: "border-slate-100",  text: "text-slate-400",  badge: "bg-slate-100 text-slate-400" },
  error:    { bg: "bg-slate-50",  border: "border-slate-100",  text: "text-slate-400",  badge: "bg-slate-100 text-slate-400" },
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
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchCalendar(); }, []);

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

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-slate-900">Earnings Calendar</h2>

      <div className="card border-amber-100 bg-amber-50">
        <p className="text-sm text-amber-700">
          <strong>Why this matters:</strong> Stocks can move +-20% on earnings day. Always check upcoming earnings before acting on an analysis.
        </p>
      </div>

      <div className="card">
        <p className="section-title mb-3">Check Specific Ticker</p>
        <form onSubmit={checkSingleTicker} className="flex gap-2">
          <TickerAutocomplete
            value={singleTicker}
            onChange={(val) => setSingleTicker(val.toUpperCase())}
            onSelect={({ symbol }) => setSingleTicker(symbol)}
            placeholder="AAPL, RELIANCE.NS..."
            showHint={false}
            className="flex-1"
          />
          <button type="submit" disabled={singleLoading || !singleTicker} className="btn-primary px-5">
            {singleLoading ? "..." : "Check"}
          </button>
        </form>

        {singleResult && !singleResult.error && (() => {
          const cfg = LEVEL_CONFIG[singleResult.warning_level] || LEVEL_CONFIG.info;
          return (
            <div className={`mt-3 p-3 rounded-xl border ${cfg.border} ${cfg.bg}`}>
              <div className={`font-semibold text-sm ${cfg.text}`}>{singleResult.emoji} {singleResult.message}</div>
              {(singleResult.eps_estimate || singleResult.revenue_estimate) && (
                <div className="mt-1.5 text-xs text-slate-500 flex gap-4">
                  {singleResult.eps_estimate && <span>EPS Est: <strong>${singleResult.eps_estimate.toFixed(2)}</strong></span>}
                  {singleResult.revenue_estimate && <span>Revenue Est: <strong>${(singleResult.revenue_estimate / 1e9).toFixed(1)}B</strong></span>}
                </div>
              )}
            </div>
          );
        })()}
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <p className="section-title">Upcoming Earnings</p>
          <button onClick={fetchCalendar} disabled={loading} className="btn-ghost text-xs">
            {loading ? "..." : "Refresh"}
          </button>
        </div>

        <div className="flex gap-2 mb-4">
          <input value={customTickers} onChange={(e) => setCustomTickers(e.target.value)} placeholder="AAPL,MSFT,GOOGL..." className="input-field text-sm flex-1" />
          <button onClick={fetchCalendar} className="btn-secondary text-sm px-4">Update</button>
        </div>

        {loading && <div className="text-center text-slate-400 py-8 text-sm">Loading...</div>}

        {calendar && !loading && (
          <div className="space-y-2">
            {calendar.upcoming_within_7_days?.length > 0 && (
              <div className="p-3 rounded-xl border border-amber-200 bg-amber-50 mb-3">
                <div className="text-amber-700 font-semibold text-xs mb-2">Earnings within 7 days:</div>
                <div className="flex flex-wrap gap-1.5">
                  {calendar.upcoming_within_7_days.map((e) => (
                    <span key={e.ticker} className="text-xs bg-amber-100 text-amber-700 px-2 py-1 rounded-lg font-mono">
                      {e.ticker} ({e.days_until_earnings}d)
                    </span>
                  ))}
                </div>
              </div>
            )}

            {calendar.calendar?.map((item) => {
              const cfg = LEVEL_CONFIG[item.warning_level] || LEVEL_CONFIG.info;
              return (
                <div key={item.ticker} className={`flex items-center justify-between p-3 rounded-xl border ${cfg.border} ${cfg.bg}`}>
                  <div className="flex items-center gap-3">
                    <span className="text-lg">{item.emoji}</span>
                    <div>
                      <div className={`font-bold font-mono text-sm ${cfg.text}`}>{item.ticker}</div>
                      <div className="text-xs text-slate-400">{item.message}</div>
                    </div>
                  </div>
                  <div className="text-right text-xs">
                    {item.next_earnings_date && <div className="font-semibold text-slate-600">{item.next_earnings_date}</div>}
                    {item.days_until_earnings !== null && item.days_until_earnings >= 0 && (
                      <div className={`font-bold ${item.days_until_earnings <= 3 ? "text-red-500" : item.days_until_earnings <= 7 ? "text-amber-500" : "text-slate-400"}`}>
                        {item.days_until_earnings === 0 ? "TODAY" : `${item.days_until_earnings}d`}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default EarningsCalendar;
