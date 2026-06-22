import { useState } from "react";
import useAnalysis from "./hooks/useAnalysis";
import TickerSearch from "./components/TickerSearch";
import SignalBadge from "./components/SignalBadge";
import BacktestChart from "./components/BacktestChart";
import MetricsTable from "./components/MetricsTable";
import RAGExplainer from "./components/RAGExplainer";
import PortfolioTracker from "./components/PortfolioTracker";
import CompareStocks from "./components/CompareStocks";
import StrategyOptimizer from "./components/StrategyOptimizer";
import PriceAlerts from "./components/PriceAlerts";
import EarningsCalendar from "./components/EarningsCalendar";
import WalkForwardAnalysis from "./components/WalkForwardAnalysis";
import LivePriceChart from "./components/LivePriceChart";

const TABS = [
  { id: "analyze",   label: "Analyze",   emoji: "🔍" },
  { id: "portfolio", label: "Portfolio", emoji: "💼" },
  { id: "compare",   label: "Compare",   emoji: "📊" },
  { id: "optimize",  label: "Optimize",  emoji: "⚙️" },
  { id: "alerts",    label: "Alerts",    emoji: "🔔" },
  { id: "earnings",  label: "Earnings",  emoji: "📅" },
  { id: "validate",  label: "Validate",  emoji: "🔬" },
  { id: "live",      label: "Live",      emoji: "📈" },
];

const HOW_IT_WORKS = [
  { emoji: "🔍", step: "Research",  desc: "Fetches real-time market data" },
  { emoji: "📚", step: "RAG",       desc: "Retrieves SEC filings and news" },
  { emoji: "🧠", step: "Sentiment", desc: "FinBERT scores news sentiment" },
  { emoji: "📊", step: "Strategy",  desc: "Selects optimal strategy" },
  { emoji: "⚡", step: "Backtest",  desc: "Runs historical simulation" },
  { emoji: "⚠️", step: "Risk",      desc: "Evaluates Sharpe, VaR, MDD" },
  { emoji: "🤖", step: "Explain",   desc: "AI generates cited answer" },
];

const AGENT_STEPS = ["Research", "RAG", "Sentiment", "Strategy", "Backtest", "Risk", "Explain"];

const App = () => {
  const [activeTab, setActiveTab] = useState("analyze");
  const { result, loading, error, analyze, reset } = useAnalysis();

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(135deg, #dde3ee 0%, #c8d0e0 100%)" }}>

      {/* ── Outer container (like the reference image's rounded white box) ── */}
      <div style={{ maxWidth: "1280px", margin: "0 auto", padding: "24px 16px" }}>

        {/* ── Main app card ── */}
        <div style={{
          background: "#f0f3f9",
          borderRadius: "24px",
          overflow: "hidden",
          boxShadow: "0 8px 40px rgba(0,0,0,0.12), 0 2px 8px rgba(0,0,0,0.08)",
          minHeight: "calc(100vh - 48px)",
        }}>

          {/* ── Header ── */}
          <div style={{
            background: "#ffffff",
            borderBottom: "1px solid #e8edf5",
            padding: "0 28px",
          }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: "60px" }}>
              {/* Logo */}
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <div style={{
                  width: "36px", height: "36px",
                  background: "linear-gradient(135deg, #4f46e5, #818cf8)",
                  borderRadius: "10px",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  boxShadow: "0 2px 8px rgba(79,70,229,0.4)",
                }}>
                  <span style={{ color: "white", fontWeight: "800", fontSize: "16px" }}>Q</span>
                </div>
                <div>
                  <div style={{ fontWeight: "800", fontSize: "16px", color: "#0f172a", letterSpacing: "-0.02em" }}>QuantMind</div>
                  <div style={{ fontSize: "11px", color: "#94a3b8" }}>AI Trading Advisor</div>
                </div>
              </div>

              {/* Right */}
              <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                <div style={{
                  display: "flex", alignItems: "center", gap: "6px",
                  fontSize: "11px", color: "#64748b",
                  background: "#f1f5f9", border: "1px solid #e2e8f0",
                  padding: "6px 12px", borderRadius: "20px",
                }}>
                  <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#10b981", display: "inline-block", animation: "pulse 2s infinite" }}></span>
                  LangGraph + RAG + DSA
                </div>
                {result && activeTab === "analyze" && (
                  <button onClick={reset} style={{ fontSize: "12px", color: "#94a3b8", background: "none", border: "none", cursor: "pointer" }}>
                    Clear
                  </button>
                )}
              </div>
            </div>

            {/* Tab navigation */}
            <div style={{ display: "flex", gap: "2px", overflowX: "auto" }}>
              {TABS.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  style={{
                    display: "flex", alignItems: "center", gap: "6px",
                    padding: "12px 16px",
                    fontSize: "13px", fontWeight: "500",
                    whiteSpace: "nowrap",
                    border: "none", background: "none", cursor: "pointer",
                    borderBottom: activeTab === tab.id ? "2px solid #4f46e5" : "2px solid transparent",
                    color: activeTab === tab.id ? "#4f46e5" : "#64748b",
                    transition: "all 0.15s",
                  }}
                >
                  <span style={{ fontSize: "14px" }}>{tab.emoji}</span>
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          {/* ── Content area ── */}
          <div style={{ padding: "24px 28px" }}>

            {/* ANALYZE TAB */}
            {activeTab === "analyze" && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "20px" }}>

                {/* Left */}
                <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                  <TickerSearch onAnalyze={analyze} loading={loading} />

                  {!result && !loading && (
                    <div className="card">
                      <p className="section-title" style={{ marginBottom: "16px" }}>How It Works</p>
                      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                        {HOW_IT_WORKS.map(({ emoji, step, desc }, i) => (
                          <div key={step} style={{ display: "flex", alignItems: "flex-start", gap: "12px" }}>
                            <div style={{
                              width: "32px", height: "32px", borderRadius: "8px",
                              background: "linear-gradient(135deg, #f1f5f9, #e2e8f0)",
                              display: "flex", alignItems: "center", justifyContent: "center",
                              flexShrink: 0, fontSize: "14px",
                            }}>
                              {emoji}
                            </div>
                            <div>
                              <div style={{ fontSize: "13px", fontWeight: "600", color: "#334155" }}>
                                <span style={{ color: "#94a3b8", marginRight: "4px" }}>{i + 1}.</span>
                                {step}
                              </div>
                              <div style={{ fontSize: "11px", color: "#94a3b8", marginTop: "2px" }}>{desc}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Right */}
                <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>

                  {error && (
                    <div className="card" style={{ borderLeft: "4px solid #ef4444", background: "#fff5f5" }}>
                      <div style={{ display: "flex", gap: "12px", alignItems: "flex-start" }}>
                        <div style={{ width: "32px", height: "32px", borderRadius: "8px", background: "#fee2e2", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                          <span style={{ color: "#dc2626", fontWeight: "700" }}>!</span>
                        </div>
                        <div>
                          <div style={{ fontWeight: "600", color: "#dc2626", fontSize: "14px" }}>Analysis Failed</div>
                          <div style={{ color: "#ef4444", fontSize: "13px", marginTop: "4px" }}>{error}</div>
                        </div>
                      </div>
                    </div>
                  )}

                  {loading && (
                    <div className="card">
                      <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
                        <div style={{
                          width: "40px", height: "40px", borderRadius: "50%",
                          border: "3px solid #e0e7ff", borderTopColor: "#4f46e5",
                          animation: "spin 0.8s linear infinite", flexShrink: 0,
                        }}></div>
                        <div>
                          <div style={{ fontWeight: "600", color: "#1e293b", fontSize: "14px" }}>Running Analysis</div>
                          <div style={{ color: "#94a3b8", fontSize: "12px", marginTop: "2px" }}>7 AI agents working in sequence</div>
                        </div>
                      </div>
                      <div style={{ marginTop: "16px", display: "flex", gap: "6px" }}>
                        {AGENT_STEPS.map((s, i) => (
                          <div key={s} style={{ flex: 1 }}>
                            <div style={{ height: "4px", borderRadius: "2px", background: "#e0e7ff", overflow: "hidden" }}>
                              <div style={{
                                height: "100%", background: "linear-gradient(90deg, #4f46e5, #818cf8)",
                                borderRadius: "2px", animation: `pulse 1.5s ease-in-out ${i * 0.15}s infinite`,
                              }} />
                            </div>
                            <div style={{ textAlign: "center", fontSize: "9px", color: "#94a3b8", marginTop: "4px" }}>{s}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {result && !loading && (
                    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                      <SignalBadge signal={result.signal} ticker={result.ticker} processingTimeMs={result.processing_time_ms} />
                      <MetricsTable backtestResults={result.backtest_results} riskMetrics={result.risk_metrics} marketData={result.market_data} />
                      <BacktestChart equityCurve={result.equity_curve} startDate={result.backtest_results?.start_date} endDate={result.backtest_results?.end_date} strategyName={result.selected_strategy} />
                      <RAGExplainer explanation={result.final_explanation} citations={result.final_citations} strategyRationale={result.strategy_rationale} selectedStrategy={result.selected_strategy} sentimentScore={result.sentiment_score} sentimentLabel={result.sentiment_label} sentimentConfidence={result.sentiment_confidence} sentimentDetails={result.sentiment_details} />
                    </div>
                  )}

                  {!result && !loading && !error && (
                    <div className="card" style={{ textAlign: "center", padding: "64px 24px", border: "2px dashed #e2e8f0" }}>
                      <div style={{
                        width: "64px", height: "64px", borderRadius: "20px",
                        background: "linear-gradient(135deg, #eef2ff, #e0e7ff)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        margin: "0 auto 16px", fontSize: "28px",
                      }}>📈</div>
                      <div style={{ fontSize: "18px", fontWeight: "700", color: "#334155", marginBottom: "8px" }}>Ready to Analyze</div>
                      <div style={{ fontSize: "13px", color: "#94a3b8", maxWidth: "280px", margin: "0 auto 20px" }}>
                        Enter a ticker and question to get an AI-powered analysis backed by SEC filings and real backtesting data.
                      </div>
                      <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "8px" }}>
                        {["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "JPM"].map((t) => (
                          <span key={t} style={{
                            fontSize: "12px", color: "#64748b",
                            background: "#f8fafc", border: "1px solid #e2e8f0",
                            padding: "6px 12px", borderRadius: "8px",
                            fontFamily: "monospace", fontWeight: "600",
                            cursor: "pointer",
                          }}>{t}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === "portfolio" && <PortfolioTracker />}
            {activeTab === "compare"   && <CompareStocks />}
            {activeTab === "optimize"  && <StrategyOptimizer />}
            {activeTab === "alerts"    && <PriceAlerts />}
            {activeTab === "earnings"  && <EarningsCalendar />}
            {activeTab === "validate"  && <WalkForwardAnalysis />}
            {activeTab === "live"      && <LivePriceChart />}

          </div>

          {/* Footer */}
          <div style={{ borderTop: "1px solid #e8edf5", padding: "16px 28px", textAlign: "center" }}>
            <span style={{ fontSize: "11px", color: "#94a3b8" }}>
              QuantMind · LangGraph + RAG + DSA ·{" "}
              <a href="https://github.com/MohanpandeyA/quantmind" target="_blank" rel="noopener noreferrer" style={{ color: "#6366f1", textDecoration: "underline" }}>
                GitHub
              </a>
            </span>
          </div>

        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 1; } }
      `}</style>
    </div>
  );
};

export default App;
