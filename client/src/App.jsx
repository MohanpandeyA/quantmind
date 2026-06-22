import { useState } from "react";
import useAnalysis from "./hooks/useAnalysis";
import { useTheme } from "./hooks/useTheme";
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

const S = {
  page: {
    minHeight: "100vh",
    background: "#EEF0F6",
    padding: "20px",
    fontFamily: '"Inter", system-ui, sans-serif',
  },
  shell: {
    maxWidth: "1320px",
    margin: "0 auto",
    background: "#F4F6FB",
    borderRadius: "20px",
    overflow: "hidden",
    boxShadow: "0 4px 24px rgba(15,23,42,0.10), 0 1px 4px rgba(15,23,42,0.06)",
    minHeight: "calc(100vh - 40px)",
    display: "flex",
    flexDirection: "column",
  },
  header: {
    background: "#FFFFFF",
    borderBottom: "1px solid #E4E8F0",
    padding: "0 28px",
    flexShrink: 0,
  },
  headerTop: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    height: "64px",
  },
  logo: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  logoIcon: {
    width: "38px",
    height: "38px",
    background: "linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)",
    borderRadius: "10px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    boxShadow: "0 2px 8px rgba(79,70,229,0.35)",
    flexShrink: 0,
  },
  logoText: {
    fontSize: "17px",
    fontWeight: "800",
    color: "#111827",
    letterSpacing: "-0.03em",
  },
  logoSub: {
    fontSize: "11px",
    color: "#9CA3AF",
    marginTop: "1px",
  },
  statusPill: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    fontSize: "11px",
    color: "#6B7280",
    background: "#F9FAFB",
    border: "1px solid #E5E7EB",
    padding: "5px 12px",
    borderRadius: "20px",
  },
  statusDot: {
    width: "7px",
    height: "7px",
    borderRadius: "50%",
    background: "#10B981",
  },
  tabs: {
    display: "flex",
    gap: "0",
    overflowX: "auto",
    scrollbarWidth: "none",
  },
  content: {
    padding: "24px 28px",
    flex: 1,
  },
  footer: {
    borderTop: "1px solid #E4E8F0",
    padding: "14px 28px",
    textAlign: "center",
    fontSize: "11px",
    color: "#9CA3AF",
    background: "#FFFFFF",
  },
};

const App = () => {
  const [activeTab, setActiveTab] = useState("analyze");
  const { result, loading, error, analyze, reset } = useAnalysis();

  return (
    <div style={S.page}>
      <div style={S.shell}>

        {/* Header */}
        <div style={S.header}>
          <div style={S.headerTop}>
            <div style={S.logo}>
              <div style={S.logoIcon}>
                <span style={{ color: "#FFFFFF", fontWeight: "800", fontSize: "17px" }}>Q</span>
              </div>
              <div>
                <div style={S.logoText}>QuantMind</div>
                <div style={S.logoSub}>AI Trading Strategy Advisor</div>
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <div style={S.statusPill}>
                <span style={{ ...S.statusDot, animation: "pulse-dot 2s ease-in-out infinite" }}></span>
                LangGraph + RAG + DSA
              </div>
              {result && activeTab === "analyze" && (
                <button onClick={reset} style={{ fontSize: "12px", color: "#9CA3AF", background: "none", border: "none", cursor: "pointer", padding: "4px 8px" }}>
                  Clear
                </button>
              )}
            </div>
          </div>

          {/* Tabs */}
          <div style={S.tabs}>
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  padding: "12px 18px",
                  fontSize: "13px",
                  fontWeight: activeTab === tab.id ? "600" : "500",
                  whiteSpace: "nowrap",
                  border: "none",
                  background: "none",
                  cursor: "pointer",
                  borderBottom: activeTab === tab.id ? "2.5px solid #4F46E5" : "2.5px solid transparent",
                  color: activeTab === tab.id ? "#4F46E5" : "#6B7280",
                  transition: "all 0.15s ease",
                  fontFamily: "inherit",
                }}
              >
                <span style={{ fontSize: "14px" }}>{tab.emoji}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div style={S.content}>

          {/* ANALYZE TAB */}
          {activeTab === "analyze" && (
            <div style={{ display: "grid", gridTemplateColumns: "340px 1fr", gap: "20px" }}>

              {/* Left column */}
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                <TickerSearch onAnalyze={analyze} loading={loading} />

                {!result && !loading && (
                  <div className="card">
                    <p className="section-title" style={{ marginBottom: "14px" }}>How It Works</p>
                    <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                      {HOW_IT_WORKS.map(({ emoji, step, desc }, i) => (
                        <div key={step} style={{ display: "flex", alignItems: "flex-start", gap: "10px" }}>
                          <div style={{
                            width: "30px", height: "30px", borderRadius: "8px",
                            background: "#F3F4F6",
                            border: "1px solid #E5E7EB",
                            display: "flex", alignItems: "center", justifyContent: "center",
                            flexShrink: 0, fontSize: "13px",
                          }}>
                            {emoji}
                          </div>
                          <div>
                            <div style={{ fontSize: "13px", fontWeight: "600", color: "#374151" }}>
                              <span style={{ color: "#D1D5DB", marginRight: "4px", fontWeight: "400" }}>{i + 1}.</span>
                              {step}
                            </div>
                            <div style={{ fontSize: "11px", color: "#9CA3AF", marginTop: "1px" }}>{desc}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Right column */}
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>

                {/* Error */}
                {error && (
                  <div style={{
                    background: "#FEF2F2", border: "1px solid #FECACA",
                    borderLeft: "4px solid #EF4444",
                    borderRadius: "12px", padding: "16px",
                    display: "flex", gap: "12px", alignItems: "flex-start",
                  }}>
                    <div style={{
                      width: "30px", height: "30px", borderRadius: "8px",
                      background: "#FEE2E2", display: "flex", alignItems: "center",
                      justifyContent: "center", flexShrink: 0,
                    }}>
                      <span style={{ color: "#DC2626", fontWeight: "700", fontSize: "14px" }}>!</span>
                    </div>
                    <div>
                      <div style={{ fontWeight: "600", color: "#991B1B", fontSize: "14px" }}>Analysis Failed</div>
                      <div style={{ color: "#B91C1C", fontSize: "13px", marginTop: "3px" }}>{error}</div>
                    </div>
                  </div>
                )}

                {/* Loading */}
                {loading && (
                  <div className="card">
                    <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
                      <div style={{
                        width: "38px", height: "38px", borderRadius: "50%",
                        border: "3px solid #E0E7FF",
                        borderTopColor: "#4F46E5",
                        animation: "spin 0.75s linear infinite",
                        flexShrink: 0,
                      }}></div>
                      <div>
                        <div style={{ fontWeight: "600", color: "#111827", fontSize: "14px" }}>Running Analysis</div>
                        <div style={{ color: "#9CA3AF", fontSize: "12px", marginTop: "2px" }}>7 AI agents working in sequence</div>
                      </div>
                    </div>
                    <div style={{ marginTop: "16px", display: "flex", gap: "4px" }}>
                      {AGENT_STEPS.map((s, i) => (
                        <div key={s} style={{ flex: 1 }}>
                          <div style={{ height: "3px", borderRadius: "2px", background: "#E0E7FF", overflow: "hidden" }}>
                            <div style={{
                              height: "100%",
                              background: "linear-gradient(90deg, #4F46E5, #818CF8)",
                              borderRadius: "2px",
                              animation: `pulse-dot 1.5s ease-in-out ${i * 0.15}s infinite`,
                            }} />
                          </div>
                          <div style={{ textAlign: "center", fontSize: "9px", color: "#9CA3AF", marginTop: "3px" }}>{s}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Results */}
                {result && !loading && (
                  <div style={{ display: "flex", flexDirection: "column", gap: "16px", animation: "fadeIn 0.3s ease-out" }}>
                    <SignalBadge signal={result.signal} ticker={result.ticker} processingTimeMs={result.processing_time_ms} />
                    <MetricsTable backtestResults={result.backtest_results} riskMetrics={result.risk_metrics} marketData={result.market_data} />
                    <BacktestChart equityCurve={result.equity_curve} startDate={result.backtest_results?.start_date} endDate={result.backtest_results?.end_date} strategyName={result.selected_strategy} />
                    <RAGExplainer explanation={result.final_explanation} citations={result.final_citations} strategyRationale={result.strategy_rationale} selectedStrategy={result.selected_strategy} sentimentScore={result.sentiment_score} sentimentLabel={result.sentiment_label} sentimentConfidence={result.sentiment_confidence} sentimentDetails={result.sentiment_details} />
                  </div>
                )}

                {/* Empty state */}
                {!result && !loading && !error && (
                  <div style={{
                    background: "#FFFFFF",
                    border: "2px dashed #E5E7EB",
                    borderRadius: "16px",
                    padding: "60px 24px",
                    textAlign: "center",
                  }}>
                    <div style={{
                      width: "60px", height: "60px", borderRadius: "16px",
                      background: "linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      margin: "0 auto 16px", fontSize: "26px",
                    }}>📈</div>
                    <div style={{ fontSize: "17px", fontWeight: "700", color: "#1F2937", marginBottom: "8px" }}>
                      Ready to Analyze
                    </div>
                    <div style={{ fontSize: "13px", color: "#9CA3AF", maxWidth: "300px", margin: "0 auto 20px", lineHeight: "1.6" }}>
                      Enter a ticker and question to get an AI-powered analysis backed by SEC filings and real backtesting data.
                    </div>
                    <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "8px" }}>
                      {["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "JPM"].map((t) => (
                        <span key={t} style={{
                          fontSize: "12px", color: "#6B7280",
                          background: "#F9FAFB",
                          border: "1px solid #E5E7EB",
                          padding: "5px 12px", borderRadius: "8px",
                          fontFamily: '"JetBrains Mono", monospace',
                          fontWeight: "600", cursor: "pointer",
                          transition: "all 0.15s",
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
        <div style={S.footer}>
          QuantMind · LangGraph + RAG + DSA ·{" "}
          <a href="https://github.com/MohanpandeyA/quantmind" target="_blank" rel="noopener noreferrer"
            style={{ color: "#6366F1", textDecoration: "underline" }}>
            GitHub
          </a>
        </div>

      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse-dot { 0%, 100% { opacity: 0.35; } 50% { opacity: 1; } }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 3px; }
      `}</style>
    </div>
  );
};

export default App;
