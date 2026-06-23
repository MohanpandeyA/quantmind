import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import TickerAutocomplete from "./TickerAutocomplete";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

const fmt = {
  usd: (v) => v != null ? `$${Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "--",
  pct: (v) => v != null ? `${Number(v) >= 0 ? "+" : ""}${(Number(v) * 100).toFixed(2)}%` : "--",
  num: (v) => v != null ? Number(v).toFixed(2) : "--",
};

const PaperTrading = () => {
  const [portfolio, setPortfolio] = useState(null);
  const [orders, setOrders] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [orderForm, setOrderForm] = useState({ ticker: "", side: "buy", qty: "1" });
  const [orderLoading, setOrderLoading] = useState(false);
  const [orderSuccess, setOrderSuccess] = useState(null);
  const [activeTab, setActiveTab] = useState("portfolio");

  const fetchPortfolio = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [portResp, ordResp] = await Promise.all([
        axios.get(`${BASE_URL}/paper-trade/portfolio`),
        axios.get(`${BASE_URL}/paper-trade/orders?limit=10`),
      ]);
      setPortfolio(portResp.data);
      setOrders(ordResp.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial fetch + auto-refresh every 10 seconds
  useEffect(() => {
    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 10_000);
    return () => clearInterval(interval);
  }, [fetchPortfolio]);

  const placeOrder = async (e) => {
    e.preventDefault();
    if (!orderForm.ticker || !orderForm.qty) return;
    setOrderLoading(true);
    setOrderSuccess(null);
    setError(null);
    try {
      const resp = await axios.post(`${BASE_URL}/paper-trade/order`, {
        ticker: orderForm.ticker.toUpperCase(),
        side: orderForm.side,
        qty: parseFloat(orderForm.qty),
      });
      setOrderSuccess(resp.data);
      setOrderForm((p) => ({ ...p, ticker: "", qty: "1" }));
      // Immediate refresh, then again after 2s for Alpaca to settle
      await fetchPortfolio();
      setTimeout(fetchPortfolio, 2000);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setOrderLoading(false);
    }
  };

  // Use unrealized_pnl (sum of open position P&L) for a more accurate display.
  // Falls back to total_pnl (equity - initial_capital) if no positions yet.
  const hasPositions = (portfolio?.positions?.length ?? 0) > 0;
  const totalPnl = hasPositions
    ? (portfolio?.unrealized_pnl ?? portfolio?.total_pnl ?? 0)
    : (portfolio?.total_pnl ?? 0);
  const totalPnlPct = portfolio?.total_pnl_pct ?? 0;

  // Detect pending (accepted but not filled) orders
  const hasPendingOrders = orders.some(
    (o) => o.status === "accepted" || o.status === "pending_new" || o.status === "new"
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <h2 style={{ fontSize: "20px", fontWeight: "700", color: "inherit", margin: 0 }}>
          Paper Trading
        </h2>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "11px", color: "#9CA3AF" }}>Alpaca Paper Account</span>
          <button onClick={fetchPortfolio} disabled={loading} className="btn-ghost" style={{ fontSize: "12px" }}>
            {loading ? "..." : "Refresh"}
          </button>
        </div>
      </div>

      <div style={{ padding: "10px 14px", background: "#FFFBEB", border: "1px solid #FDE68A", borderRadius: "10px" }}>
        <p style={{ fontSize: "12px", color: "#92400E", margin: 0 }}>
          Paper trading uses virtual money ($100,000 starting balance). No real money is at risk.
        </p>
      </div>

      {hasPendingOrders && (
        <div style={{ padding: "10px 14px", background: "#EFF6FF", border: "1px solid #BFDBFE", borderRadius: "10px" }}>
          <p style={{ fontSize: "12px", color: "#1E40AF", margin: 0 }}>
            ⏳ <strong>Orders pending fill</strong> — Alpaca paper trading fills market orders during US market hours
            (Mon–Fri 9:30 AM – 4:00 PM ET). P&L will update once orders are filled.
          </p>
        </div>
      )}

      {error && (
        <div className="card" style={{ borderLeft: "4px solid #EF4444", background: "#FEF2F2" }}>
          <p style={{ color: "#DC2626", fontSize: "13px", margin: 0 }}>{error}</p>
        </div>
      )}

      {portfolio && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" }}>
          <div className="metric-card">
            <div className="metric-value">{fmt.usd(portfolio.equity)}</div>
            <div className="metric-label">Total Equity</div>
          </div>
          <div className="metric-card">
            <div className="metric-value" style={{ color: totalPnl >= 0 ? "#059669" : "#DC2626" }}>
              {fmt.usd(totalPnl)}
            </div>
            <div className="metric-label">Total P&L</div>
          </div>
          <div className="metric-card">
            <div className="metric-value" style={{ color: totalPnlPct >= 0 ? "#059669" : "#DC2626" }}>
              {fmt.pct(totalPnlPct)}
            </div>
            <div className="metric-label">Return</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{fmt.usd(portfolio.buying_power)}</div>
            <div className="metric-label">Buying Power</div>
          </div>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "340px 1fr", gap: "16px" }}>
        <div className="card">
          <p className="section-title" style={{ marginBottom: "14px" }}>Place Order</p>
          <form onSubmit={placeOrder} style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            <div>
              <label className="section-title" style={{ display: "block", marginBottom: "6px" }}>Ticker</label>
              <TickerAutocomplete
                value={orderForm.ticker}
                onChange={(v) => setOrderForm((p) => ({ ...p, ticker: v.toUpperCase() }))}
                onSelect={({ symbol }) => setOrderForm((p) => ({ ...p, ticker: symbol }))}
                placeholder="AAPL"
                showHint={false}
              />
            </div>
            <div>
              <label className="section-title" style={{ display: "block", marginBottom: "6px" }}>Side</label>
              <div style={{ display: "flex", gap: "8px" }}>
                {["buy", "sell"].map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setOrderForm((p) => ({ ...p, side: s }))}
                    style={{
                      flex: 1, padding: "8px", fontSize: "13px", fontWeight: "600",
                      borderRadius: "8px", border: "1.5px solid",
                      cursor: "pointer", fontFamily: "inherit",
                      background: orderForm.side === s ? (s === "buy" ? "#059669" : "#DC2626") : "transparent",
                      color: orderForm.side === s ? "#FFFFFF" : (s === "buy" ? "#059669" : "#DC2626"),
                      borderColor: s === "buy" ? "#059669" : "#DC2626",
                    }}
                  >
                    {s === "buy" ? "BUY" : "SELL"}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="section-title" style={{ display: "block", marginBottom: "6px" }}>Quantity (shares)</label>
              <input
                type="number"
                min="1"
                step="1"
                value={orderForm.qty}
                onChange={(e) => setOrderForm((p) => ({ ...p, qty: e.target.value }))}
                className="input-field"
                required
              />
            </div>
            <button type="submit" disabled={orderLoading || !orderForm.ticker} className="btn-primary">
              {orderLoading ? "Placing..." : `Place ${orderForm.side.toUpperCase()} Order`}
            </button>
          </form>

          {orderSuccess && (
            <div style={{ marginTop: "12px", padding: "10px 12px", background: "#ECFDF5", border: "1px solid #A7F3D0", borderRadius: "8px" }}>
              <p style={{ fontSize: "12px", color: "#065F46", fontWeight: "600", margin: "0 0 2px" }}>
                Order placed!
              </p>
              <p style={{ fontSize: "11px", color: "#059669", margin: 0 }}>
                {orderSuccess.side?.toUpperCase()} {orderSuccess.qty} {orderSuccess.ticker} — Status: {orderSuccess.status}
              </p>
            </div>
          )}
        </div>

        <div className="card">
          <div style={{ display: "flex", gap: "12px", marginBottom: "14px" }}>
            {["portfolio", "orders"].map((t) => (
              <button
                key={t}
                onClick={() => setActiveTab(t)}
                style={{
                  padding: "6px 14px", fontSize: "12px", fontWeight: "600",
                  borderRadius: "8px", border: "none", cursor: "pointer",
                  background: activeTab === t ? "#4F46E5" : "transparent",
                  color: activeTab === t ? "#FFFFFF" : "#9CA3AF",
                  fontFamily: "inherit",
                }}
              >
                {t === "portfolio" ? "Positions" : "Order History"}
              </button>
            ))}
          </div>

          {activeTab === "portfolio" && (
            <>
              {!portfolio?.positions?.length ? (
                <div style={{ textAlign: "center", padding: "32px", color: "#9CA3AF", fontSize: "13px" }}>
                  No open positions. Place your first order.
                </div>
              ) : (
                portfolio.positions.map((p) => (
                  <div key={p.ticker} className="table-row">
                    <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                      <div style={{ width: "36px", height: "36px", borderRadius: "8px", background: "#EEF2FF", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <span style={{ fontSize: "11px", fontWeight: "700", color: "#4F46E5" }}>{p.ticker?.slice(0, 2)}</span>
                      </div>
                      <div>
                        <div style={{ fontWeight: "700", fontSize: "13px", fontFamily: "monospace" }}>{p.ticker}</div>
                        <div style={{ fontSize: "11px", color: "#9CA3AF" }}>{p.qty} shares @ {fmt.usd(p.avg_entry_price)}</div>
                      </div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontWeight: "600", fontSize: "13px" }}>{fmt.usd(p.market_value)}</div>
                      <div style={{ fontSize: "11px", color: p.unrealized_pl >= 0 ? "#059669" : "#DC2626", fontWeight: "600" }}>
                        {fmt.usd(p.unrealized_pl)} ({fmt.pct(p.unrealized_plpc)})
                      </div>
                    </div>
                  </div>
                ))
              )}
            </>
          )}

          {activeTab === "orders" && (
            <>
              {!orders.length ? (
                <div style={{ textAlign: "center", padding: "32px", color: "#9CA3AF", fontSize: "13px" }}>
                  No orders yet.
                </div>
              ) : (
                orders.map((o) => (
                  <div key={o.order_id} className="table-row">
                    <div>
                      <div style={{ fontWeight: "600", fontSize: "13px", fontFamily: "monospace" }}>
                        <span style={{ color: o.side === "buy" ? "#059669" : "#DC2626", marginRight: "6px" }}>
                          {o.side?.toUpperCase()}
                        </span>
                        {o.ticker}
                      </div>
                      <div style={{ fontSize: "11px", color: "#9CA3AF" }}>
                        {o.qty} shares {o.filled_avg_price ? `@ ${fmt.usd(o.filled_avg_price)}` : ""}
                      </div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <span style={{
                        fontSize: "11px", fontWeight: "600", padding: "2px 8px", borderRadius: "20px",
                        background: o.status === "filled" ? "#ECFDF5" : o.status === "canceled" ? "#FEF2F2" : "#FFFBEB",
                        color: o.status === "filled" ? "#065F46" : o.status === "canceled" ? "#991B1B" : "#92400E",
                      }}>
                        {o.status}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default PaperTrading;
