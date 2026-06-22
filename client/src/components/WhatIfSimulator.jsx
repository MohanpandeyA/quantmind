import { useState } from "react";
import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

const QUICK_SCENARIOS = [
  { label: "Drop 10%", question: "What if the stock drops 10% tomorrow?" },
  { label: "Drop 20%", question: "What if the stock drops 20%?" },
  { label: "Rise 15%", question: "What if the stock rises 15%?" },
  { label: "COVID crash", question: "What would happen during the 2020 COVID crash?" },
  { label: "2022 bear", question: "What if we were in the 2022 bear market?" },
  { label: "2008 crisis", question: "What would happen during the 2008 financial crisis?" },
];

const WhatIfSimulator = () => {
  const [ticker, setTicker] = useState("AAPL");
  const [question, setQuestion] = useState("");
  const [strategy, setStrategy] = useState("momentum");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runSimulation = async (q) => {
    const finalQ = q || question;
    if (!finalQ.trim() || !ticker.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const resp = await axios.post(`${BASE_URL}/whatif`, {
        ticker: ticker.toUpperCase(),
        question: finalQ,
        strategy,
      });
      setResult(resp.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const signalColor = (sig) => {
    if (sig === "BUY") return "#059669";
    if (sig === "SELL") return "#DC2626";
    return "#D97706";
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <h2 style={{ fontSize: "20px", fontWeight: "700", color: "inherit", margin: 0 }}>
        What-If Simulator
      </h2>

      <div className="card">
        <p className="section-title" style={{ marginBottom: "14px" }}>Scenario Configuration</p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "12px" }}>
          <div>
            <label className="section-title" style={{ display: "block", marginBottom: "6px" }}>Ticker</label>
            <input
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="AAPL"
              className="input-field"
              style={{ fontFamily: "monospace", fontWeight: "600" }}
            />
          </div>
          <div>
            <label className="section-title" style={{ display: "block", marginBottom: "6px" }}>Strategy</label>
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)} className="input-field">
              <option value="momentum">Momentum</option>
              <option value="mean_reversion">Mean Reversion</option>
              <option value="rsi">RSI</option>
              <option value="macd">MACD</option>
            </select>
          </div>
        </div>

        <div style={{ marginBottom: "12px" }}>
          <label className="section-title" style={{ display: "block", marginBottom: "6px" }}>Quick Scenarios</label>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
            {QUICK_SCENARIOS.map((s) => (
              <button
                key={s.label}
                onClick={() => { setQuestion(s.question); runSimulation(s.question); }}
                disabled={loading}
                style={{
                  padding: "6px 12px", fontSize: "12px", fontWeight: "500",
                  background: "#EEF2FF", color: "#4F46E5",
                  border: "1px solid #C7D2FE", borderRadius: "8px",
                  cursor: "pointer", fontFamily: "inherit",
                  transition: "all 0.15s",
                }}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        <div style={{ marginBottom: "12px" }}>
          <label className="section-title" style={{ display: "block", marginBottom: "6px" }}>Or Ask Anything</label>
          <div style={{ display: "flex", gap: "8px" }}>
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What if AAPL drops 10% tomorrow?"
              className="input-field"
              onKeyDown={(e) => e.key === "Enter" && runSimulation()}
            />
            <button
              onClick={() => runSimulation()}
              disabled={loading || !question.trim()}
              className="btn-primary"
              style={{ whiteSpace: "nowrap" }}
            >
              {loading ? "..." : "Simulate"}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="card" style={{ borderLeft: "4px solid #EF4444", background: "#FEF2F2" }}>
          <p style={{ color: "#DC2626", fontSize: "13px", margin: 0 }}>{error}</p>
        </div>
      )}

      {loading && (
        <div className="card" style={{ textAlign: "center", padding: "32px" }}>
          <div style={{ width: "36px", height: "36px", borderRadius: "50%", border: "3px solid #E0E7FF", borderTopColor: "#4F46E5", animation: "spin 0.75s linear infinite", margin: "0 auto 12px" }}></div>
          <p style={{ color: "#9CA3AF", fontSize: "13px", margin: 0 }}>Running scenario analysis...</p>
        </div>
      )}

      {result && !loading && (
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          <div className="card" style={{ borderLeft: "4px solid #4F46E5" }}>
            <p className="section-title" style={{ marginBottom: "10px" }}>Scenario Result</p>
            <p style={{ fontSize: "13px", color: "#6B7280", marginBottom: "12px", fontStyle: "italic" }}>
              "{result.question}"
            </p>
            <p style={{ fontSize: "12px", color: "#9CA3AF", marginBottom: "12px" }}>
              {result.scenario_description}
            </p>

            <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
              {result.original_signal && (
                <div className="metric-card" style={{ flex: 1, minWidth: "120px" }}>
                  <div style={{ fontSize: "11px", color: "#9CA3AF", marginBottom: "4px" }}>Original Signal</div>
                  <div style={{ fontSize: "20px", fontWeight: "700", color: signalColor(result.original_signal) }}>
                    {result.original_signal}
                  </div>
                </div>
              )}
              {result.new_signal && (
                <div className="metric-card" style={{ flex: 1, minWidth: "120px" }}>
                  <div style={{ fontSize: "11px", color: "#9CA3AF", marginBottom: "4px" }}>New Signal</div>
                  <div style={{ fontSize: "20px", fontWeight: "700", color: signalColor(result.new_signal) }}>
                    {result.new_signal}
                  </div>
                </div>
              )}
              {result.price_impact != null && (
                <div className="metric-card" style={{ flex: 1, minWidth: "120px" }}>
                  <div style={{ fontSize: "11px", color: "#9CA3AF", marginBottom: "4px" }}>Price Impact</div>
                  <div style={{ fontSize: "20px", fontWeight: "700", color: result.price_impact >= 0 ? "#059669" : "#DC2626" }}>
                    {result.price_impact >= 0 ? "+" : ""}{result.price_impact?.toFixed(1)}%
                  </div>
                </div>
              )}
              {result.new_sharpe != null && (
                <div className="metric-card" style={{ flex: 1, minWidth: "120px" }}>
                  <div style={{ fontSize: "11px", color: "#9CA3AF", marginBottom: "4px" }}>New Sharpe</div>
                  <div style={{ fontSize: "20px", fontWeight: "700", color: result.new_sharpe >= 0.5 ? "#059669" : "#DC2626" }}>
                    {result.new_sharpe?.toFixed(2)}
                  </div>
                </div>
              )}
              {result.historical_occurrences > 0 && (
                <div className="metric-card" style={{ flex: 1, minWidth: "120px" }}>
                  <div style={{ fontSize: "11px", color: "#9CA3AF", marginBottom: "4px" }}>Occurrences (5yr)</div>
                  <div style={{ fontSize: "20px", fontWeight: "700", color: "#4F46E5" }}>
                    {result.historical_occurrences}x
                  </div>
                </div>
              )}
              {result.avg_recovery_days && (
                <div className="metric-card" style={{ flex: 1, minWidth: "120px" }}>
                  <div style={{ fontSize: "11px", color: "#9CA3AF", marginBottom: "4px" }}>Avg Recovery</div>
                  <div style={{ fontSize: "20px", fontWeight: "700", color: "#6B7280" }}>
                    {result.avg_recovery_days}d
                  </div>
                </div>
              )}
            </div>

            {result.signal_change && (
              <div style={{ marginTop: "12px", padding: "8px 12px", background: "#FFFBEB", border: "1px solid #FDE68A", borderRadius: "8px" }}>
                <span style={{ fontSize: "12px", color: "#92400E", fontWeight: "600" }}>
                  Signal change: {result.signal_change}
                </span>
              </div>
            )}
          </div>

          <div className="card">
            <p className="section-title" style={{ marginBottom: "10px" }}>AI Analysis</p>
            <p style={{ fontSize: "13px", color: "#374151", lineHeight: "1.7", margin: 0 }}>
              {result.ai_explanation}
            </p>
            <p style={{ fontSize: "11px", color: "#9CA3AF", marginTop: "8px", margin: "8px 0 0" }}>
              Processed in {result.processing_time_ms?.toFixed(0)}ms
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default WhatIfSimulator;
