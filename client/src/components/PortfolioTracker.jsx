import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import TickerAutocomplete from "./TickerAutocomplete";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

const fmt = {
  price: (v) => v != null ? `$${Number(v).toFixed(2)}` : "--",
  pct: (v) => v != null ? `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(2)}%` : "--",
  usd: (v) => v != null ? `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 })}` : "--",
};

const PortfolioTracker = () => {
  const [performance, setPerformance] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newPosition, setNewPosition] = useState({
    ticker: "", shares: "", entry_price: "",
    entry_date: new Date().toISOString().split("T")[0], notes: "",
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
      setShowAddForm(false);
      setNewPosition({ ticker: "", shares: "", entry_price: "", entry_date: new Date().toISOString().split("T")[0], notes: "" });
      fetchPerformance();
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  };

  const removePosition = async (ticker) => {
    try {
      await axios.delete(`${BASE_URL}/portfolio/positions/${ticker}`);
      fetchPerformance();
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  };

  const totalPnl = performance?.total_unrealized_pnl || 0;
  const totalPnlPct = performance?.total_unrealized_pnl_pct || 0;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-slate-900">Portfolio</h2>
        <button onClick={() => setShowAddForm(!showAddForm)} className="btn-primary text-sm">
          {showAddForm ? "Cancel" : "+ Add Position"}
        </button>
      </div>

      {error && (
        <div className="card border-red-100 bg-red-50">
          <p className="text-red-600 text-sm">{error}</p>
        </div>
      )}

      {/* Summary cards */}
      {performance && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="metric-card">
            <div className="metric-value">{fmt.usd(performance.total_current_value)}</div>
            <div className="metric-label">Total Value</div>
          </div>
          <div className="metric-card">
            <div className={`metric-value ${totalPnl >= 0 ? "text-emerald-600" : "text-red-600"}`}>
              {fmt.usd(totalPnl)}
            </div>
            <div className="metric-label">Unrealized P&L</div>
          </div>
          <div className="metric-card">
            <div className={`metric-value ${totalPnlPct >= 0 ? "text-emerald-600" : "text-red-600"}`}>
              {fmt.pct(totalPnlPct)}
            </div>
            <div className="metric-label">Return</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{performance.position_count}</div>
            <div className="metric-label">Positions</div>
          </div>
        </div>
      )}

      {/* Add position form */}
      {showAddForm && (
        <div className="card">
          <p className="section-title mb-4">Add Position</p>
          <form onSubmit={addPosition} className="grid grid-cols-2 gap-3">
            <div className="col-span-2 sm:col-span-1">
              <label className="text-xs font-medium text-slate-500 mb-1 block">Ticker</label>
              <TickerAutocomplete
                value={newPosition.ticker}
                onChange={(v) => setNewPosition((p) => ({ ...p, ticker: v.toUpperCase() }))}
                onSelect={({ symbol }) => setNewPosition((p) => ({ ...p, ticker: symbol }))}
                placeholder="AAPL"
                showHint={false}
              />
            </div>
            <div>
              <label className="text-xs font-medium text-slate-500 mb-1 block">Shares</label>
              <input type="number" value={newPosition.shares} onChange={(e) => setNewPosition((p) => ({ ...p, shares: e.target.value }))} placeholder="10" className="input-field" required />
            </div>
            <div>
              <label className="text-xs font-medium text-slate-500 mb-1 block">Entry Price</label>
              <input type="number" step="0.01" value={newPosition.entry_price} onChange={(e) => setNewPosition((p) => ({ ...p, entry_price: e.target.value }))} placeholder="150.00" className="input-field" required />
            </div>
            <div>
              <label className="text-xs font-medium text-slate-500 mb-1 block">Entry Date</label>
              <input type="date" value={newPosition.entry_date} onChange={(e) => setNewPosition((p) => ({ ...p, entry_date: e.target.value }))} className="input-field" />
            </div>
            <div className="col-span-2">
              <button type="submit" className="btn-primary w-full">Add Position</button>
            </div>
          </form>
        </div>
      )}

      {/* Positions table */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <p className="section-title">Positions</p>
          <button onClick={fetchPerformance} disabled={loading} className="btn-ghost text-xs">
            {loading ? "..." : "Refresh"}
          </button>
        </div>

        {loading && !performance && (
          <div className="space-y-2">
            {[1, 2, 3].map((i) => <div key={i} className="skeleton h-12 rounded-xl" />)}
          </div>
        )}

        {performance?.positions?.length === 0 && (
          <div className="text-center py-10">
            <div className="w-12 h-12 bg-slate-50 rounded-xl flex items-center justify-center mx-auto mb-3">
              <span className="text-xl">💼</span>
            </div>
            <p className="text-slate-500 text-sm">No positions yet</p>
            <p className="text-slate-400 text-xs mt-1">Add your first position to track P&L</p>
          </div>
        )}

        {performance?.positions?.map((pos) => {
          const pnl = pos.unrealized_pnl || 0;
          const pnlPct = pos.unrealized_pnl_pct || 0;
          return (
            <div key={pos.ticker} className="table-row">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 bg-indigo-50 rounded-lg flex items-center justify-center">
                  <span className="text-xs font-bold text-indigo-600">{pos.ticker?.slice(0, 2)}</span>
                </div>
                <div>
                  <div className="font-semibold text-slate-900 text-sm font-mono">{pos.ticker}</div>
                  <div className="text-xs text-slate-400">{pos.shares} shares @ {fmt.price(pos.entry_price)}</div>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="text-sm font-semibold text-slate-900">{fmt.price(pos.current_price)}</div>
                  <div className={`text-xs font-medium ${pnl >= 0 ? "text-emerald-600" : "text-red-600"}`}>
                    {fmt.usd(pnl)} ({fmt.pct(pnlPct)})
                  </div>
                </div>
                <button onClick={() => removePosition(pos.ticker)} className="btn-ghost text-xs text-red-400 hover:text-red-600 hover:bg-red-50">
                  Remove
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default PortfolioTracker;
