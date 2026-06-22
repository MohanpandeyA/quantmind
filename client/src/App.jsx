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
  { id: "analyze",   label: "Analyze"   },
  { id: "portfolio", label: "Portfolio" },
  { id: "compare",   label: "Compare"   },
  { id: "optimize",  label: "Optimize"  },
  { id: "alerts",    label: "Alerts"    },
  { id: "earnings",  label: "Earnings"  },
  { id: "validate",  label: "Validate"  },
  { id: "live",      label: "Live"      },
];

const HOW_IT_WORKS = [
  { step: "Research",  desc: "Fetches real-time market data" },
  { step: "RAG",       desc: "Retrieves SEC filings and news" },
  { step: "Sentiment", desc: "FinBERT scores news sentiment" },
  { step: "Strategy",  desc: "Selects optimal strategy" },
  { step: "Backtest",  desc: "Runs historical simulation" },
  { step: "Risk",      desc: "Evaluates Sharpe, VaR, MDD" },
  { step: "Explain",   desc: "AI generates cited answer" },
];

const AGENT_STEPS = ["Research", "RAG", "Sentiment", "Strategy", "Backtest", "Risk", "Explain"];

const App = () => {
  const [activeTab, setActiveTab] = useState("analyze");
  const { result, loading, error, analyze, reset } = useAnalysis();
  const { theme, toggleTheme } = useTheme();

  const isDark = theme === "dark";

  const colors = {
    page:       isDark ? "#0F1117"  : "#EEF0F6",
    shell:      isDark ? "#1A1F2E"  : "#F4F6FB",
    header:     isDark ? "#111827"  : "#FFFFFF",
    headerBorder: isDark ? "#1F2937" : "#E4E8F0",
    card:       isDark ? "#1E2433"  : "#FFFFFF",
    cardBorder: isDark ? "#2D3748"  : "#E4E8F0",
    text:       isDark ? "#F1F5F9"  : "#111827",
    textSub:    isDark ? "#9CA3AF"  : "#6B7280",
    textMuted:  isDark ? "#6B7280"  : "#9CA3AF",
    inputBg:    isDark ? "#252D3D"  : "#FFFFFF",
    inputBorder:isDark ? "#374151"  : "#D1D5DB",
    metricBg:   isDark ? "#252D3D"  : "#F9FAFB",
    metricBorder:isDark ? "#374151" : "#E5E7EB",
    tabActive:  "#4F46E5",
    tabInactive:isDark ? "#6B7280"  : "#6B7280",
    emptyBorder:isDark ? "#2D3748"  : "#E5E7EB",
    emptyBg:    isDark ? "#1E2433"  : "#FFFFFF",
    toggleBg:   isDark ? "#252D3D"  : "#F3F4F6",
    toggleBorder:isDark ? "#374151" : "#E5E7EB",
    footerBg:   isDark ? "#111827"  : "#FFFFFF",
    footerBorder:isDark ? "#1F2937" : "#E4E8F0",
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: isDark
        ? "linear-gradient(135deg, #0F1117 0%, #161B27 100%)"
        : "linear-gradient(135deg, #EEF0F6 0%, #DDE3EE 100%)",
      padding: "20px",
      fontFamily: '"Inter", system-ui, sans-serif',
      transition: "background 0.3s ease",
    }}>
      <div style={{
        maxWidth: "1320px",
        margin: "0 auto",
        background: colors.shell,
        borderRadius: "20px",
        overflow: "hidden",
        boxShadow: isDark
          ? "0 4px 24px rgba(0,0,0,0.5), 0 1px 4px rgba(0,0,0,0.4)"
          : "0 4px 24px rgba(15,23,42,0.10), 0 1px 4px rgba(15,23,42,0.06)",
        minHeight: "calc(100vh - 40px)",
        display: "flex",
        flexDirection: "column",
        transition: "all 0.3s ease",
      }}>

        {/* Header */}
        <div style={{
          background: colors.header,
          borderBottom: `1px solid ${colors.headerBorder}`,
          padding: "0 28px",
          transition: "all 0.3s ease",
        }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: "60px" }}>

            {/* Logo */}
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <div style={{
                width: "36px", height: "36px",
                background: "linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)",
                borderRadius: "10px",
                display: "flex", alignItems: "center", justifyContent: "center",
                boxShadow: "0 2px 8px rgba(79,70,229,0.35)",
                flexShrink: 0,
              }}>
                <span style={{ color: "#FFFFFF", fontWeight: "800", fontSize: "16px" }}>Q</span>
              </div>
              <div>
                <div style={{ fontWeight: "800", fontSize: "16px", color: colors.text, letterSpacing: "-0.03em" }}>
                  QuantMind
                </div>
                <div style={{ fontSize: "11px", color: colors.textMuted }}>
                  AI Trading Advisor
                </div>
              </div>
            </div>

            {/* Right side — dark mode toggle only */}
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              {result && activeTab === "analyze" && (
                <button onClick={reset} style={{
                  fontSize: "12px", color: colors.textMuted,
                  background: "none", border: "none", cursor: "pointer",
                  padding: "4px 8px", fontFamily: "inherit",
                }}>
                  Clear
                </button>
              )}

              {/* Dark/Light toggle */}
              <button
                onClick={toggleTheme}
                title={isDark ? "Switch to light mode" : "Switch to dark mode"}
                style={{
                  width: "38px", height: "38px",
                  borderRadius: "10px",
                  border: `1px solid ${colors.toggleBorder}`,
                  background: colors.toggleBg,
                  cursor: "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: "17px",
                  transition: "all 0.2s ease",
                  flexShrink: 0,
                }}
              >
                {isDark ? "☀️" : "🌙"}
              </button>
            </div>
          </div>

          {/* Tab navigation — clean text only, no emojis */}
          <div style={{ display: "flex", overflowX: "auto", scrollbarWidth: "none" }}>
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  padding: "12px 18px",
                  fontSize: "13px",
                  fontWeight: activeTab === tab.id ? "600" : "500",
                  whiteSpace: "nowrap",
                  border: "none",
                  background: "none",
                  cursor: "pointer",
                  borderBottom: activeTab === tab.id ? "2.5px solid #4F46E5" : "2.5px solid transparent",
                  color: activeTab === tab.id ? "#4F46E5" : colors.tabInactive,
                  transition: "all 0.15s ease",
                  fontFamily: "inherit",
                  letterSpacing: "-0.01em",
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div style={{ padding: "24px 28px", flex: 1 }}>

          {/* ANALYZE TAB */}
          {activeTab === "analyze" && (
            <div style={{ display: "grid", gridTemplateColumns: "340px 1fr", gap: "20px" }}>

              {/* Left */}
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                <TickerSearch onAnalyze={analyze} loading={loading} />

                {!result && !loading && (
                  <div style={{
                    background: colors.card,
                    border: `1px solid ${colors.cardBorder}`,
                    borderRadius: "16px",
                    padding: "20px",
                    boxShadow: isDark ? "0 1px 3px rgba(0,0,0,0.3)" : "0 1px 3px rgba(16,24,40,0.06)",
                  }}>
                    <p style={{ fontSize: "11px", fontWeight: "700", color: colors.textMuted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "14px" }}>
                      How It Works
                    </p>
                    <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                      {HOW_IT_WORKS.map(({ step, desc }, i) => (
                        <div key={step} style={{ display: "flex", alignItems: "flex-start", gap: "10px" }}>
                          <div style={{
                            width: "22px", height: "22px", borderRadius: "6px",
                            background: "linear-gradient(135deg, #4F46E5, #7C3AED)",
                            display: "flex", alignItems: "center", justifyContent: "center",
                            flexShrink: 0, fontSize: "10px", fontWeight: "700", color: "#FFFFFF",
                          }}>
                            {i + 1}
                          </div>
                          <div>
                            <div style={{ fontSize: "13px", fontWeight: "600", color: colors.text }}>{step}</div>
                            <div style={{ fontSize: "11px", color: colors.textMuted, marginTop: "1px" }}>{desc}</div>
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
                  <div style={{
                    background: isDark ? "#2D1515" : "#FEF2F2",
                    border: `1px solid ${isDark ? "#7F1D1D" : "#FECACA"}`,
                    borderLeft: "4px solid #EF4444",
                    borderRadius: "12px", padding: "16px",
                    display: "flex", gap: "12px", alignItems: "flex-start",
                  }}>
                    <div style={{ width: "28px", height: "28px", borderRadius: "8px", background: isDark ? "#7F1D1D" : "#FEE2E2", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                      <span style={{ color: "#EF4444", fontWeight: "700", fontSize: "13px" }}>!</span>
                    </div>
                    <div>
                      <div style={{ fontWeight: "600", color: "#EF4444", fontSize: "14px" }}>Analysis Failed</div>
                      <div style={{ color: isDark ? "#FCA5A5" : "#B91C1C", fontSize: "13px", marginTop: "3px" }}>{error}</div>
                    </div>
                  </div>
                )}

                {loading && (
                  <div style={{ background: colors.card, border: `1px solid ${colors.cardBorder}`, borderRadius: "16px", padding: "20px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
                      <div style={{ width: "36px", height: "36px", borderRadius: "50%", border: "3px solid #E0E7FF", borderTopColor: "#4F46E5", animation: "spin 0.75s linear infinite", flexShrink: 0 }}></div>
                      <div>
                        <div style={{ fontWeight: "600", color: colors.text, fontSize: "14px" }}>Running Analysis</div>
                        <div style={{ color: colors.textMuted, fontSize: "12px", marginTop: "2px" }}>7 AI agents working in sequence</div>
                      </div>
                    </div>
                    <div style={{ marginTop: "16px", display: "flex", gap: "4px" }}>
                      {AGENT_STEPS.map((s, i) => (
                        <div key={s} style={{ flex: 1 }}>
                          <div style={{ height: "3px", borderRadius: "2px", background: isDark ? "#2D3748" : "#E0E7FF", overflow: "hidden" }}>
                            <div style={{ height: "100%", background: "linear-gradient(90deg, #4F46E5, #818CF8)", borderRadius: "2px", animation: `pulse-dot 1.5s ease-in-out ${i * 0.15}s infinite` }} />
                          </div>
                          <div style={{ textAlign: "center", fontSize: "9px", color: colors.textMuted, marginTop: "3px" }}>{s}</div>
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
                  <div style={{ background: colors.emptyBg, border: `2px dashed ${colors.emptyBorder}`, borderRadius: "16px", padding: "60px 24px", textAlign: "center" }}>
                    <div style={{ width: "56px", height: "56px", borderRadius: "16px", background: isDark ? "#1E2433" : "linear-gradient(135deg, #EEF2FF, #E0E7FF)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px", fontSize: "24px" }}>📈</div>
                    <div style={{ fontSize: "17px", fontWeight: "700", color: colors.text, marginBottom: "8px" }}>Ready to Analyze</div>
                    <div style={{ fontSize: "13px", color: colors.textMuted, maxWidth: "300px", margin: "0 auto 20px", lineHeight: "1.6" }}>
                      Enter a ticker and question to get an AI-powered analysis backed by SEC filings and real backtesting data.
                    </div>
                    <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "8px" }}>
                      {["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "JPM"].map((t) => (
                        <span key={t} style={{ fontSize: "12px", color: colors.textSub, background: isDark ? "#252D3D" : "#F9FAFB", border: `1px solid ${colors.cardBorder}`, padding: "5px 12px", borderRadius: "8px", fontFamily: '"JetBrains Mono", monospace', fontWeight: "600", cursor: "pointer" }}>{t}</span>
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
        <div style={{ borderTop: `1px solid ${colors.footerBorder}`, padding: "14px 28px", textAlign: "center", fontSize: "11px", color: colors.textMuted, background: colors.footerBg, transition: "all 0.3s ease" }}>
          QuantMind &middot; LangGraph + RAG + DSA &middot;{" "}
          <a href="https://github.com/MohanpandeyA/quantmind" target="_blank" rel="noopener noreferrer" style={{ color: "#6366F1", textDecoration: "underline" }}>
            GitHub
          </a>
        </div>

      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
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
