/**
 * PriceAlerts — real-time price alert notifications via WebSocket.
 *
 * WHY WEBSOCKET:
 *   HTTP polling wastes bandwidth and adds latency.
 *   WebSocket keeps a persistent connection — alerts arrive instantly
 *   when a price threshold is crossed. This is how Bloomberg works.
 *
 * ALERT CONDITIONS:
 *   - price_below: "Alert me when AAPL drops below $200" (buy opportunity)
 *   - price_above: "Alert me when AAPL rises above $250" (take profit)
 *   - change_pct_below: "Alert me when AAPL falls 5% today" (stop loss)
 *   - change_pct_above: "Alert me when AAPL gains 5% today" (momentum)
 *
 * CONNECTION STATUS:
 *   🟢 Connected — WebSocket active, alerts monitoring
 *   🔴 Disconnected — WebSocket closed, auto-reconnect in 5s
 *   🟡 Connecting — WebSocket handshake in progress
 */

import { useState, useEffect, useRef, useCallback } from "react";
import TickerAutocomplete from "./TickerAutocomplete";

const WS_URL = "ws://localhost:8000/alerts/ws";

const CONDITION_LABELS = {
  price_below: "Price drops below $",
  price_above: "Price rises above $",
  change_pct_below: "Daily change drops below %",
  change_pct_above: "Daily change rises above %",
};

const PriceAlerts = () => {
  const [status, setStatus] = useState("disconnected"); // connected | disconnected | connecting
  const [alerts, setAlerts] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [priceUpdates, setPriceUpdates] = useState({});
  const [newAlert, setNewAlert] = useState({
    ticker: "",
    condition: "price_below",
    threshold: "",
    message: "",
  });
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      clearTimeout(reconnectTimer.current);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === "connected") {
          setAlerts(msg.alerts || []);
        } else if (msg.type === "alert") {
          // Price alert triggered!
          setNotifications((prev) => [
            { ...msg, id: Date.now() },
            ...prev.slice(0, 19), // Keep last 20
          ]);
          // Mark alert as triggered
          setAlerts((prev) =>
            prev.map((a) =>
              a.ticker === msg.ticker && a.condition === msg.condition
                ? { ...a, triggered: true }
                : a
            )
          );
        } else if (msg.type === "price_update") {
          setPriceUpdates(msg.prices || {});
        } else if (msg.type === "alert_added") {
          setAlerts((prev) => [
            ...prev.filter(
              (a) => !(a.ticker === msg.ticker && a.condition === msg.condition)
            ),
            { ticker: msg.ticker, condition: msg.condition, threshold: msg.threshold, triggered: false },
          ]);
        } else if (msg.type === "alert_removed") {
          setAlerts((prev) => prev.filter((a) => a.ticker !== msg.ticker));
        }
      } catch (e) {
        console.error("WebSocket message parse error:", e);
      }
    };

    ws.onclose = () => {
      setStatus("disconnected");
      // Auto-reconnect after 5 seconds
      reconnectTimer.current = setTimeout(connect, 5000);
    };

    ws.onerror = () => {
      setStatus("disconnected");
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendMessage = (msg) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  };

  const addAlert = (e) => {
    e.preventDefault();
    if (!newAlert.ticker || !newAlert.threshold) return;
    sendMessage({
      action: "add",
      ticker: newAlert.ticker.toUpperCase(),
      condition: newAlert.condition,
      threshold: parseFloat(newAlert.threshold),
      message: newAlert.message,
    });
    setNewAlert({ ticker: "", condition: "price_below", threshold: "", message: "" });
  };

  const removeAlert = (ticker, condition) => {
    sendMessage({ action: "remove", ticker, condition });
  };

  const dismissNotification = (id) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  };

  const statusConfig = {
    connected: { color: "text-green-400", dot: "bg-green-500", label: "Connected" },
    connecting: { color: "text-yellow-400", dot: "bg-yellow-500 animate-pulse", label: "Connecting..." },
    disconnected: { color: "text-red-400", dot: "bg-red-500", label: "Disconnected (reconnecting in 5s)" },
  };
  const sc = statusConfig[status];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <span>🔔</span> Price Alerts
        </h2>
        <div className={`flex items-center gap-2 text-sm ${sc.color}`}>
          <span className={`w-2 h-2 rounded-full ${sc.dot}`}></span>
          {sc.label}
        </div>
      </div>

      {/* Triggered notifications */}
      {notifications.length > 0 && (
        <div className="space-y-2">
          {notifications.slice(0, 5).map((n) => (
            <div key={n.id} className="card border border-orange-700 bg-orange-900/20 flex items-start justify-between gap-3">
              <div>
                <div className="text-orange-400 font-semibold text-sm">{n.message}</div>
                <div className="text-xs text-gray-500 mt-1">{new Date(n.timestamp).toLocaleTimeString()}</div>
              </div>
              <button onClick={() => dismissNotification(n.id)} className="text-gray-500 hover:text-gray-300 shrink-0">✕</button>
            </div>
          ))}
        </div>
      )}

      {/* Add alert form */}
      <div className="card">
        <h3 className="text-sm font-semibold text-gray-400 mb-3">Add New Alert</h3>
        <form onSubmit={addAlert} className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Ticker</label>
              <TickerAutocomplete
                value={newAlert.ticker}
                onChange={(val) => setNewAlert({ ...newAlert, ticker: val.toUpperCase() })}
                onSelect={({ symbol }) => setNewAlert({ ...newAlert, ticker: symbol })}
                placeholder="AAPL, RELIANCE.NS..."
                disabled={status !== "connected"}
                showHint={false}
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Condition</label>
              <select
                value={newAlert.condition}
                onChange={(e) => setNewAlert({ ...newAlert, condition: e.target.value })}
                className="input-field"
                disabled={status !== "connected"}
              >
                {Object.entries(CONDITION_LABELS).map(([k, v]) => (
                  <option key={k} value={k}>{v}...</option>
                ))}
              </select>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Threshold ({newAlert.condition.includes("pct") ? "%" : "$"})
              </label>
              <input
                type="number"
                value={newAlert.threshold}
                onChange={(e) => setNewAlert({ ...newAlert, threshold: e.target.value })}
                placeholder={newAlert.condition.includes("pct") ? "-5" : "200.00"}
                className="input-field"
                step="0.01"
                disabled={status !== "connected"}
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Note (optional)</label>
              <input
                value={newAlert.message}
                onChange={(e) => setNewAlert({ ...newAlert, message: e.target.value })}
                placeholder="Buy opportunity"
                className="input-field"
                disabled={status !== "connected"}
              />
            </div>
          </div>
          <button
            type="submit"
            disabled={status !== "connected" || !newAlert.ticker || !newAlert.threshold}
            className="btn-primary w-full text-sm"
          >
            🔔 Set Alert
          </button>
        </form>
      </div>

      {/* Active alerts */}
      {alerts.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-400 mb-3">
            Active Alerts ({alerts.filter((a) => !a.triggered).length} watching)
          </h3>
          <div className="space-y-2">
            {alerts.map((alert, i) => {
              const priceData = priceUpdates[alert.ticker];
              return (
                <div
                  key={i}
                  className={`flex items-center justify-between p-3 rounded-lg border ${
                    alert.triggered
                      ? "border-gray-700 bg-gray-800/20 opacity-50"
                      : "border-gray-700 bg-gray-800/30"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span className={`text-lg ${alert.triggered ? "grayscale" : ""}`}>
                      {alert.triggered ? "✅" : "🔔"}
                    </span>
                    <div>
                      <div className="text-white font-mono font-semibold">{alert.ticker}</div>
                      <div className="text-xs text-gray-400">
                        {CONDITION_LABELS[alert.condition]}{alert.threshold}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    {priceData && (
                      <div className="text-right text-sm">
                        <div className="text-white">${priceData.current_price?.toFixed(2)}</div>
                        <div className={priceData.price_change_pct >= 0 ? "text-green-400 text-xs" : "text-red-400 text-xs"}>
                          {priceData.price_change_pct >= 0 ? "+" : ""}{priceData.price_change_pct?.toFixed(2)}%
                        </div>
                      </div>
                    )}
                    {!alert.triggered && (
                      <button
                        onClick={() => removeAlert(alert.ticker, alert.condition)}
                        className="text-gray-600 hover:text-red-400 transition-colors"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {alerts.length === 0 && notifications.length === 0 && (
        <div className="card border-dashed border-gray-700 text-center py-10">
          <div className="text-4xl mb-3">🔔</div>
          <p className="text-gray-400 text-sm">No alerts set. Add one above to get notified when prices cross your thresholds.</p>
          <p className="text-gray-600 text-xs mt-2">Prices checked every 30 seconds via WebSocket</p>
        </div>
      )}
    </div>
  );
};

export default PriceAlerts;
