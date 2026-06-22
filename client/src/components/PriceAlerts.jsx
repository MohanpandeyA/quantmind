import { useState, useEffect, useRef } from "react";
import TickerAutocomplete from "./TickerAutocomplete";

const _apiBase = import.meta.env.VITE_API_URL || "";
const _wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
const WS_URL = _apiBase.startsWith("http")
  ? _apiBase.replace(/^http/, "ws") + "/alerts/ws"
  : `${_wsProto}//${window.location.hostname}:8003/alerts/ws`;

const PriceAlerts = () => {
  const [connected, setConnected] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [fired, setFired] = useState([]);
  const [ticker, setTicker] = useState("");
  const [condition, setCondition] = useState("price_below");
  const [threshold, setThreshold] = useState("");
  const wsRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === "connected") setAlerts(msg.alerts || []);
        else if (msg.type === "alert_added") setAlerts((prev) => [...prev, { ticker: msg.ticker, condition: msg.condition, threshold: msg.threshold }]);
        else if (msg.type === "alert_removed") setAlerts((prev) => prev.filter((a) => a.ticker !== msg.ticker));
        else if (msg.type === "alert") setFired((prev) => [{ ...msg, time: new Date().toLocaleTimeString() }, ...prev.slice(0, 9)]);
      } catch {}
    };

    return () => ws.close();
  }, []);

  const send = (msg) => wsRef.current?.readyState === 1 && wsRef.current.send(JSON.stringify(msg));

  const addAlert = (e) => {
    e.preventDefault();
    if (!ticker || !threshold) return;
    send({ action: "add", ticker: ticker.toUpperCase(), condition, threshold: parseFloat(threshold) });
    setTicker("");
    setThreshold("");
  };

  const removeAlert = (t) => send({ action: "remove", ticker: t });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-slate-900">Price Alerts</h2>
        <div className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border ${connected ? "bg-emerald-50 border-emerald-100 text-emerald-700" : "bg-slate-50 border-slate-200 text-slate-500"}`}>
          <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-emerald-500 animate-pulse" : "bg-slate-400"}`}></span>
          {connected ? "Connected" : "Disconnected"}
        </div>
      </div>

      <div className="card">
        <p className="section-title mb-3">Add Alert</p>
        <form onSubmit={addAlert} className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="col-span-2 sm:col-span-1">
              <label className="text-xs font-medium text-slate-500 mb-1 block">Ticker</label>
              <TickerAutocomplete value={ticker} onChange={(v) => setTicker(v.toUpperCase())} onSelect={({ symbol }) => setTicker(symbol)} placeholder="AAPL" showHint={false} />
            </div>
            <div>
              <label className="text-xs font-medium text-slate-500 mb-1 block">Condition</label>
              <select value={condition} onChange={(e) => setCondition(e.target.value)} className="input-field">
                <option value="price_below">Price Below</option>
                <option value="price_above">Price Above</option>
                <option value="change_pct_above">Change % Above</option>
                <option value="change_pct_below">Change % Below</option>
              </select>
            </div>
            <div>
              <label className="text-xs font-medium text-slate-500 mb-1 block">Threshold</label>
              <input type="number" step="0.01" value={threshold} onChange={(e) => setThreshold(e.target.value)} placeholder="200.00" className="input-field" required />
            </div>
          </div>
          <button type="submit" disabled={!connected || !ticker || !threshold} className="btn-primary w-full">
            Set Alert
          </button>
        </form>
      </div>

      {alerts.length > 0 && (
        <div className="card">
          <p className="section-title mb-3">Active Alerts ({alerts.length})</p>
          <div className="space-y-2">
            {alerts.map((a, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-slate-50 rounded-xl border border-slate-100">
                <div>
                  <span className="font-mono font-bold text-slate-900 text-sm">{a.ticker}</span>
                  <span className="text-slate-400 text-xs ml-2">{a.condition?.replace(/_/g, " ")} {a.threshold}</span>
                </div>
                <button onClick={() => removeAlert(a.ticker)} className="btn-ghost text-xs text-red-400 hover:text-red-600">Remove</button>
              </div>
            ))}
          </div>
        </div>
      )}

      {fired.length > 0 && (
        <div className="card">
          <p className="section-title mb-3">Fired Alerts</p>
          <div className="space-y-2">
            {fired.map((f, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-amber-50 rounded-xl border border-amber-100">
                <div>
                  <span className="font-mono font-bold text-amber-700 text-sm">{f.ticker}</span>
                  <span className="text-amber-600 text-xs ml-2">{f.message}</span>
                </div>
                <span className="text-xs text-slate-400">{f.time}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PriceAlerts;
