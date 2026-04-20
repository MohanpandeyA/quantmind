/**
 * QuantMind Dashboard — main React application with 6-tab navigation.
 *
 * Tabs:
 *   1. Analyze    — single stock AI analysis (original feature)
 *   2. Portfolio  — real-time P&L tracker (HV Feature 1)
 *   3. Compare    — rank multiple tickers (HV Feature 2)
 *   4. Optimize   — find best strategy params (HV Feature 3)
 *   5. Alerts     — WebSocket price alerts (HV Feature 4)
 *   6. Earnings   — upcoming earnings calendar (HV Feature 5)
 */

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

const TABS = [
  { id: "analyze",   label: "🔍 Analyze",  desc: "Single stock AI analysis" },
  { id: "portfolio", label: "💼 Portfolio", desc: "Real-time P&L tracker" },
  { id: "compare",   label: "📊 Compare",  desc: "Rank multiple tickers" },
  { id: "optimize",  label: "⚙️ Optimize", desc: "Find best parameters" },
  { id: "alerts",    label: "🔔 Alerts",   desc: "WebSocket price alerts" },
  { id: "earnings",  label: "📅 Earnings", desc: "Upcoming earnings calendar" },
];

const App = () => {
  const [activeTab, setActiveTab] = useState("analyze");
  const { result, loading, error, analyze, reset } = useAnalysis();

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">⚡</span>
            <div>
              <h1 className="text-xl font-black text-white tracking-tight">QuantMind</h1>
              <p className="text-xs text-gray-500">AI-Powered Trading Strategy Advisor</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
              LangGraph + RAG + DSA
            </span>
            {result && activeTab === "analyze" && (
              <button onClick={reset} className="text-gray-400 hover:text-gray-300 ml-2">✕ Clear</button>
            )}
          </div>
        </div>

        {/* Tab navigation */}
        <div className="max-w-7xl mx-auto px-4 flex gap-1 pb-0 overflow-x-auto">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3 py-2 text-sm font-medium rounded-t-lg transition-colors whitespace-nowrap ${
                activeTab === tab.id
                  ? "bg-gray-950 text-white border-t border-l border-r border-gray-800"
                  : "text-gray-500 hover:text-gray-300"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 py-6">

        {/* ===== TAB 1: ANALYZE ===== */}
        {activeTab === "analyze" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1 space-y-6">
              <TickerSearch onAnalyze={analyze} loading={loading} />
              {!result && !loading && (
                <div className="card">
                  <h3 className="text-sm font-semibold text-gray-400 mb-3 uppercase tracking-wider">How It Works</h3>
                  <div className="space-y-3">
                    {[
                      { icon: "🔍", step: "1. Research", desc: "Fetches real-time market data" },
                      { icon: "📚", step: "2. RAG", desc: "Retrieves SEC filings & news" },
                      { icon: "📊", step: "3. Strategy", desc: "Selects optimal strategy" },
                      { icon: "⚡", step: "4. Backtest", desc: "Runs historical simulation" },
                      { icon: "⚠️", step: "5. Risk", desc: "Evaluates Sharpe, VaR, MDD" },
                      { icon: "🤖", step: "6. Explain", desc: "AI generates cited answer" },
                    ].map(({ icon, step, desc }) => (
                      <div key={step} className="flex items-start gap-3">
                        <span className="text-lg">{icon}</span>
                        <div>
                          <div className="text-sm font-medium text-gray-300">{step}</div>
                          <div className="text-xs text-gray-500">{desc}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="lg:col-span-2 space-y-6">
              {error && (
                <div className="card border border-red-800 bg-red-900/20">
                  <div className="flex items-start gap-3">
                    <span className="text-2xl">❌</span>
                    <div>
                      <h3 className="text-red-400 font-semibold">Analysis Failed</h3>
                      <p className="text-red-300/80 text-sm mt-1">{error}</p>
                    </div>
                  </div>
                </div>
              )}

              {loading && (
                <div className="card border border-blue-800/50">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-full border-4 border-blue-900 border-t-blue-400 animate-spin"></div>
                    <div>
                      <h3 className="text-blue-400 font-semibold">Running Analysis...</h3>
                      <p className="text-gray-400 text-sm">LangGraph agents working. ~10-15 seconds.</p>
                    </div>
                  </div>
                </div>
              )}

              {result && !loading && (
                <>
                  <SignalBadge signal={result.signal} ticker={result.ticker} processingTimeMs={result.processing_time_ms} />
                  <BacktestChart equityCurve={result.equity_curve} startDate={result.backtest_results?.start_date} endDate={result.backtest_results?.end_date} strategyName={result.selected_strategy} />
                  <MetricsTable backtestResults={result.backtest_results} riskMetrics={result.risk_metrics} marketData={result.market_data} />
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
                </>
              )}

              {!result && !loading && !error && (
                <div className="card border-dashed border-gray-700 text-center py-16">
                  <div className="text-5xl mb-4">📈</div>
                  <h3 className="text-xl font-semibold text-gray-400 mb-2">Ready to Analyze</h3>
                  <p className="text-gray-500 text-sm max-w-sm mx-auto">
                    Enter a ticker and question to get an AI-powered analysis backed by SEC filings and real backtesting data.
                  </p>
                  <div className="mt-6 flex flex-wrap justify-center gap-2">
                    {["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "JPM"].map((t) => (
                      <span key={t} className="text-xs text-gray-500 bg-gray-800 px-3 py-1 rounded-full">{t}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ===== TAB 2: PORTFOLIO ===== */}
        {activeTab === "portfolio" && <PortfolioTracker />}

        {/* ===== TAB 3: COMPARE ===== */}
        {activeTab === "compare" && <CompareStocks />}

        {/* ===== TAB 4: OPTIMIZE ===== */}
        {activeTab === "optimize" && <StrategyOptimizer />}

        {/* ===== TAB 5: ALERTS ===== */}
        {activeTab === "alerts" && <PriceAlerts />}

        {/* ===== TAB 6: EARNINGS ===== */}
        {activeTab === "earnings" && <EarningsCalendar />}

      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-16 py-6 text-center text-xs text-gray-600">
        QuantMind — LangGraph + RAG + DSA + MERN |{" "}
        <a href="https://github.com/MohanpandeyA/quantmind" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-gray-400 underline">
          GitHub
        </a>
      </footer>
    </div>
  );
};

export default App;
