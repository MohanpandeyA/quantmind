/**
 * PortfolioTracker — real-time P&L for all portfolio positions.
 *
 * WHY THIS COMPONENT EXISTS:
 *   A trader manages multiple positions simultaneously. This component:
 *   1. Shows all positions with real-time P&L (green/red)
 *   2. Displays portfolio totals (total value, total return)
 *   3. Highlights best and worst performers
 *   4. Allows adding/removing positions
 *
 * DESIGN DECISIONS:
 *   - Color coding: green = profit, red = loss (universal trading convention)
 *   - Percentage AND absolute P&L shown (traders need both)
 *   - 52-week range bar shows where current price sits in the year's range
 *   - "Add Position" form is inline (no modal) for faster workflow
 */

import { useState, useEffect, useCallback } from "react";
import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

const PortfolioTracker = () => {
  const [performance, setPerformance] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newPosition, setNewPosition] = useState({
    ticker: "",
    shares: "",
    entry_price: "",
    entry_date: new Date().toISOString().split("T")[0],
    notes: "",
  });

  const fetchPerformance = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await axios.get(`${BASE_URL}/portfolio/performance`);
      setPerformance(resp.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPerformance();
    // Auto-refresh every 60 seconds
    const interval = setInterval(fetchPerformance, 60_000);
    return () => clearInterval(interval);
  }, [fetchPerformance]);

  const addPosition = async (e) => {
    e.preventDefault();
    try {
      await axios.post(`${BASE_URL}/portfolio/positions`, {
        ...newPosition,
        shares: parseFloat(newPosition.shares),
        entry_price: parseFloat(newPosition.entry_price),
      });
      setNewPosition({ ticker: "", shares: "", entry_price: "", entry_date: new Date().toISOString().split("T")[0], notes: "" });
      setShowAddForm(false);
      fetchPerformance();
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  };

  const removePosition = async (ticker) => {
    if (!confirm(`Remove ${ticker} from portfolio?`)) return;
    try {
      await axios.delete(`${BASE_URL}/portfolio/positions/${ticker}`);
      fetchPerformance();
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  };

  const pnlColor = (val) => val >= 0 ? "text-green-400" : "text-red-400";
  const pnlBg = (val) => val >= 0 ? "bg-green-900/20 border-green-800" : "bg-red-900/20 border-red-800";
  const fmt = {
    price: (v) => `$${v?.toFixed(2) ?? "—"}`,
    pct: (v) => `${v >= 0 ? "+" : ""}${v?.toFixed(2) ?? "0"}%`,
    usd: (v) => `${v >= 0 ? "+" : ""}$${Math.abs(v ?? 0).toFixed(2)}`,
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <span>💼</span> Portfolio Tracker
        </h2>
        <div className="flex gap-2">
          <button
            onClick={fetchPerformance}
            className="text-xs text-gray-400 hover:text-gray-300 bg-gray-800 px-3 py-1.5 rounded-lg"
          >
            🔄 Refresh
          </button>
          <button
            onClick={() => setShowAddForm(!showAddForm)}
            className="text-xs text-blue-400 hover:text-blue-300 bg-blue-900/30 px-3 py-1.5 rounded-lg"
          >
            + Add Position
          </button>
        </div>
      </div>

      {/* Add Position Form */}
      {showAddForm && (
        <div className="card border border-blue-800/50">
          <h3 className="text-sm font-semibold text-blue-400 mb-3">Add New Position</h3>
          <form onSubmit={addPosition} className="grid grid-cols-2 gap-3">
            <input
              placeholder="Ticker (e.g. AAPL)"
              value={newPosition.ticker}
              onChange={(e) => setNewPosition({ ...newPosition, ticker: e.target.value.toUpperCase() })}
              className="input-field text-sm"
              required
            />
            <input
              type="number"
              placeholder="Shares"
              value={newPosition.shares}
              onChange={(e) => setNewPosition({ ...newPosition, shares: e.target.value })}
              className="input-field text-sm"
              min="0.001"
              step="0.001"
              required
            />
            <input
              type="number"
              placeholder="Entry Price ($)"
              value={newPosition.entry_price}
              onChange={(e) => setNewPosition({ ...newPosition, entry_price: e.target.value })}
              className="input-field text-sm"
              min="0.01"
              step="0.01"
              required
            />
            <input
              type="date"
              value={newPosition.entry_date}
              onChange={(e) => setNewPosition({ ...newPosition, entry_date: e.target.value })}
              className="input-field text-sm"
              required
            />
            <input
              placeholder="Notes (optional)"
              value={newPosition.notes}
              onChange={(e) => setNewPosition({ ...newPosition, notes: e.target.value })}
              className="input-field text-sm col-span-2"
            />
            <div className="col-span-2 flex gap-2">
              <button type="submit" className="btn-primary text-sm flex-1">Add Position</button>
              <button type="button" onClick={() => setShowAddForm(false)} className="text-sm text-gray-400 hover:text-gray-300 px-4">Cancel</button>
            </div>
          </form>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="card border border-red-800 bg-red-900/20 text-red-400 text-sm">{error}</div>
      )}

      {/* Loading */}
      {loading && !performance && (
        <div className="card text-center text-gray-500 py-8">Loading portfolio...</div>
      )}

      {/* Empty state */}
      {performance && performance.position_count === 0 && (
        <div className="card border-dashed border-gray-700 text-center py-12">
          <div className="text-4xl mb-3">💼</div>
          <p className="text-gray-400">No positions yet. Add your first position above.</p>
        </div>
      )}

      {/* Portfolio summary */}
      {performance && performance.position_count > 0 && (
        <>
          <div className={`card border ${pnlBg(performance.total_unrealized_pnl)}`}>
            <div className="grid grid-cols-4 gap-4">
              <div>
                <div className="metric-label">Total Value</div>
                <div className="metric-value text-white">{fmt.price(performance.total_current_value)}</div>
              </div>
              <div>
                <div className="metric-label">Cost Basis</div>
                <div className="metric-value text-gray-300">{fmt.price(performance.total_cost_basis)}</div>
              </div>
              <div>
                <div className="metric-label">Unrealized P&L</div>
                <div className={`metric-value ${pnlColor(performance.total_unrealized_pnl)}`}>
                  {fmt.usd(performance.total_unrealized_pnl)}
                </div>
              </div>
              <div>
                <div className="metric-label">Total Return</div>
                <div className={`metric-value ${pnlColor(performance.total_unrealized_pnl_pct)}`}>
                  {fmt.pct(performance.total_unrealized_pnl_pct)}
                </div>
              </div>
            </div>
            {performance.best_performer && (
              <div className="mt-3 pt-3 border-t border-gray-800 flex gap-4 text-xs text-gray-400">
                <span>🏆 Best: <span className="text-green-400 font-semibold">{performance.best_performer}</span></span>
                <span>📉 Worst: <span className="text-red-400 font-semibold">{performance.worst_performer}</span></span>
                <span className="ml-auto">Updated: {new Date(performance.last_updated).toLocaleTimeString()}</span>
              </div>
            )}
          </div>

          {/* Position rows */}
          <div className="space-y-2">
            {performance.positions.map((pos) => (
              <div key={pos.ticker} className={`card border ${pnlBg(pos.unrealized_pnl)} hover:border-opacity-80 transition-all`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div>
                      <div className="text-lg font-bold text-white font-mono">{pos.ticker}</div>
                      <div className="text-xs text-gray-500">{pos.shares} shares @ {fmt.price(pos.entry_price)}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-white font-semibold">{fmt.price(pos.current_price)}</div>
                      <div className={`text-xs ${pnlColor(pos.price_change_pct)}`}>
                        {fmt.pct(pos.price_change_pct)} today
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-6">
                    <div className="text-right">
                      <div className={`font-bold ${pnlColor(pos.unrealized_pnl)}`}>
                        {fmt.usd(pos.unrealized_pnl)}
                      </div>
                      <div className={`text-sm ${pnlColor(pos.unrealized_pnl_pct)}`}>
                        {fmt.pct(pos.unrealized_pnl_pct)}
                      </div>
                    </div>
                    <div className="text-right text-xs text-gray-500">
                      <div>Value: {fmt.price(pos.current_value)}</div>
                      <div>Cost: {fmt.price(pos.cost_basis)}</div>
                    </div>
                    <button
                      onClick={() => removePosition(pos.ticker)}
                      className="text-gray-600 hover:text-red-400 transition-colors text-lg"
                      title="Remove position"
                    >
                      ✕
                    </button>
                  </div>
                </div>

                {/* 52-week range bar */}
                <div className="mt-2 pt-2 border-t border-gray-800/50">
                  <div className="flex items-center gap-2 text-xs text-gray-500">
                    <span>{fmt.price(pos.week_52_low)}</span>
                    <div className="flex-1 bg-gray-800 rounded-full h-1.5 relative">
                      <div
                        className="absolute top-0 left-0 h-full bg-blue-500 rounded-full"
                        style={{
                          width: `${Math.min(100, Math.max(0,
                            ((pos.current_price - pos.week_52_low) /
                             (pos.week_52_high - pos.week_52_low)) * 100
                          ))}%`
                        }}
                      />
                    </div>
                    <span>{fmt.price(pos.week_52_high)}</span>
                    <span className="text-gray-600">52W</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default PortfolioTracker;
