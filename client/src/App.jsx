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
  { id: "analyze",   label: "Analyze",   icon: "Search" },
  { id: "portfolio", label: "Portfolio", icon: "Briefcase" },
  { id: "compare",   label: "Compare",   icon: "BarChart2" },
  { id: "optimize",  label: "Optimize",  icon: "Settings" },
  { id: "alerts",    label: "Alerts",    icon: "Bell" },
  { id: "earnings",  label: "Earnings",  icon: "Calendar" },
  { id: "validate",  label: "Validate",  icon: "FlaskConical" },
  { id: "live",      label: "Live",      icon: "TrendingUp" },
];

const TAB_EMOJIS = {
  analyze: "🔍", portfolio: "💼", compare: "📊", optimize: "⚙️",
  alerts: "🔔", earnings: "📅", validate: "🔬", live: "📈",
};

const HOW_IT_WORKS = [
  { icon: "🔍", step: "Research",  desc: "Fetches real-time market data" },
  { icon: "📚", step: "RAG",       desc: "Retrieves SEC filings and news" },
  { icon: "🧠", step: "Sentiment", desc: "FinBERT scores news sentiment" },
  { icon: "📊", step: "Strategy",  desc: "Selects optimal strategy" },
  { icon: "⚡", step: "Backtest",  desc: "Runs historical simulation" },
  { icon: "⚠️", step: "Risk",      desc: "Evaluates Sharpe, VaR, MDD" },
  { icon: "🤖", step: "Explain",   desc: "AI generates cited answer" },
];

const AGENT_STEPS = ["Research", "RAG", "Sentiment", "Strategy", "Backtest", "Risk", "Explain"];

const App = () => {
  const [activeTab, setActiveTab] = useState("analyze");
  const { result, loading, error, analyze, reset } = useAnalysis();

  return (
    <div className="min-h-screen bg-slate-50">

      {/* Header */}
      <header className="bg-white border-b border-slate-100 sticky top-0 z-20 shadow-sm">
        <div className="max-w-7xl mx-auto px-6">

          {/* Top bar */}
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center shadow-sm">
                <span className="text-white text-sm font-bold">Q</span>
              </div>
              <div>
                <span className="text-slate-900 font-bold text-base tracking-tight">QuantMind</span>
                <span className="hidden sm:inline text-slate-400 text-xs ml-2">AI Trading Advisor</span>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-400 bg-slate-50 border border-slate-100 px-3 py-1.5 rounded-full">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                LangGraph + RAG + DSA
              </div>
              {result && activeTab === "analyze" && (
                <button onClick={reset} className="btn-ghost text-xs">
                  Clear
                </button>
              )}
            </div>
          </div>

          {/* Tab navigation */}
          <div className="flex gap-0.5 overflow-x-auto -mb-px">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={[
                  "flex items-center gap-1.5 px-4 py-3 text-sm font-medium whitespace-nowrap border-b-2 transition-all duration-150",
                  activeTab === tab.id
                    ? "border-indigo-600 text-indigo-600"
                    : "border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-200",
                ].join(" ")}
              >
                <span className="text-base leading-none">{TAB_EMOJIS[tab.id]}</span>
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-6 py-6">

        {/* ANALYZE TAB */}
        {activeTab === "analyze" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

            {/* Left column */}
            <div className="lg:col-span-1 space-y-4">
              <TickerSearch onAnalyze={analyze} loading={loading} />

              {!result && !loading && (
                <div className="card">
                  <p className="section-title mb-4">How It Works</p>
                  <div className="space-y-3">
                    {HOW_IT_WORKS.map(({ icon, step, desc }, i) => (
                      <div key={step} className="flex items-start gap-3">
                        <div className="w-7 h-7 rounded-lg bg-slate-50 border border-slate-100 flex items-center justify-center flex-shrink-0 text-sm">
                          {icon}
                        </div>
                        <div>
                          <div className="text-sm font-medium text-slate-700">
                            <span className="text-slate-400 mr-1">{i + 1}.</span>
                            {step}
                          </div>
                          <div className="text-xs text-slate-400 mt-0.5">{desc}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Right column */}
            <div className="lg:col-span-2 space-y-4">

              {/* Error state */}
              {error && (
                <div className="card border-red-100 bg-red-50 animate-fade-in">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-lg bg-red-100 flex items-center justify-center flex-shrink-0">
                      <span className="text-red-600 text-sm font-bold">!</span>
                    </div>
                    <div>
                      <h3 className="text-red-700 font-semibold text-sm">Analysis Failed</h3>
                      <p className="text-red-600/80 text-sm mt-1">{error}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Loading state */}
              {loading && (
                <div className="card animate-fade-in">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full border-2 border-slate-100 border-t-indigo-600 animate-spin flex-shrink-0"></div>
                    <div>
                      <h3 className="text-slate-800 font-semibold text-sm">Running Analysis</h3>
                      <p className="text-slate-400 text-xs mt-0.5">7 AI agents working in sequence</p>
                    </div>
                  </div>
                  <div className="mt-4 flex gap-1.5">
                    {AGENT_STEPS.map((s, i) => (
                      <div key={s} className="flex-1">
                        <div className="h-1 rounded-full bg-slate-100 overflow-hidden">
                          <div
                            className="h-full bg-indigo-500 rounded-full animate-pulse"
                            style={{ animationDelay: `${i * 0.15}s` }}
                          />
                        </div>
                        <p className="text-center text-slate-400 mt-1" style={{ fontSize: "9px" }}>{s}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Results */}
              {result && !loading && (
                <div className="space-y-4 animate-slide-up">
                  <SignalBadge
                    signal={result.signal}
                    ticker={result.ticker}
                    processingTimeMs={result.processing_time_ms}
                  />
                  <MetricsTable
                    backtestResults={result.backtest_results}
                    riskMetrics={result.risk_metrics}
                    marketData={result.market_data}
                  />
                  <BacktestChart
                    equityCurve={result.equity_curve}
                    startDate={result.backtest_results?.start_date}
                    endDate={result.backtest_results?.end_date}
                    strategyName={result.selected_strategy}
                  />
                  <RAGExplainer
                    explanation={result.final_explanation}
                    citations={result.final_citations}
                    strategyRationale={result.strategy_rationale}
                    selectedStrategy={result.selected_strategy}
                    sentimentScore={result.sentiment_score}
                    sentimentLabel={result.sentiment_label}
                    sentimentConfidence={result.sentiment_confidence}
                    sentimentDetails={result.sentiment_details}
                  />
                </div>
              )}

              {/* Empty state */}
              {!result && !loading && !error && (
                <div className="card border-dashed border-slate-200 text-center py-16 animate-fade-in">
                  <div className="w-14 h-14 bg-indigo-50 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <span className="text-2xl">📈</span>
                  </div>
                  <h3 className="text-lg font-semibold text-slate-700 mb-1">Ready to Analyze</h3>
                  <p className="text-slate-400 text-sm max-w-xs mx-auto">
                    Enter a ticker and question to get an AI-powered analysis backed by SEC filings and real backtesting data.
                  </p>
                  <div className="mt-5 flex flex-wrap justify-center gap-2">
                    {["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "JPM"].map((t) => (
                      <span
                        key={t}
                        className="text-xs text-slate-500 bg-white border border-slate-200 px-3 py-1.5 rounded-lg font-mono font-medium hover:border-indigo-300 hover:text-indigo-600 cursor-pointer transition-colors"
                      >
                        {t}
                      </span>
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

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-100 mt-12 py-5 text-center">
        <p className="text-xs text-slate-400">
          QuantMind · LangGraph + RAG + DSA ·{" "}
          <a
            href="https://github.com/MohanpandeyA/quantmind"
            target="_blank"
            rel="noopener noreferrer"
            className="text-indigo-500 hover:text-indigo-600 underline underline-offset-2"
          >
            GitHub
          </a>
        </p>
      </footer>
    </div>
  );
};

export default App;
