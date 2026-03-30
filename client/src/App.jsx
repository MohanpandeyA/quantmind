/**
 * QuantMind Dashboard — main React application.
 *
 * Layout:
 * ┌─────────────────────────────────────────────┐
 * │  Header: QuantMind logo + tagline            │
 * ├──────────────┬──────────────────────────────┤
 * │ TickerSearch │ SignalBadge (result)          │
 * │              ├──────────────────────────────┤
 * │              │ BacktestChart (equity curve)  │
 * │              ├──────────────────────────────┤
 * │              │ MetricsTable (Sharpe/VaR/...) │
 * │              ├──────────────────────────────┤
 * │              │ RAGExplainer (AI + citations) │
 * └──────────────┴──────────────────────────────┘
 */

import useAnalysis from "./hooks/useAnalysis";
import TickerSearch from "./components/TickerSearch";
import SignalBadge from "./components/SignalBadge";
import BacktestChart from "./components/BacktestChart";
import MetricsTable from "./components/MetricsTable";
import RAGExplainer from "./components/RAGExplainer";

const App = () => {
  const { result, loading, error, analyze, reset } = useAnalysis();

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">⚡</span>
            <div>
              <h1 className="text-xl font-black text-white tracking-tight">
                QuantMind
              </h1>
              <p className="text-xs text-gray-500">
                AI-Powered Trading Strategy Advisor
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4 text-xs text-gray-500">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
              LangGraph + RAG + DSA
            </span>
            {result && (
              <button
                onClick={reset}
                className="text-gray-400 hover:text-gray-300 transition-colors"
              >
                ✕ Clear
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left column: Search */}
          <div className="lg:col-span-1 space-y-6">
            <TickerSearch onAnalyze={analyze} loading={loading} />

            {/* How it works */}
            {!result && !loading && (
              <div className="card">
                <h3 className="text-sm font-semibold text-gray-400 mb-3 uppercase tracking-wider">
                  How It Works
                </h3>
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

          {/* Right column: Results */}
          <div className="lg:col-span-2 space-y-6">
            {/* Error state */}
            {error && (
              <div className="card border border-red-800 bg-red-900/20">
                <div className="flex items-start gap-3">
                  <span className="text-2xl">❌</span>
                  <div>
                    <h3 className="text-red-400 font-semibold">Analysis Failed</h3>
                    <p className="text-red-300/80 text-sm mt-1">{error}</p>
                    {error.includes("not running") && (
                      <div className="mt-3 bg-gray-900 rounded-lg p-3 text-xs font-mono text-gray-400">
                        <p className="text-gray-300 mb-1">Start the backend:</p>
                        <p>cd quantmind/backend</p>
                        <p>source .venv/bin/activate</p>
                        <p>uvicorn api.main:app --port 8000</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Loading state */}
            {loading && (
              <div className="card border border-blue-800/50">
                <div className="flex items-center gap-4">
                  <div className="relative">
                    <div className="w-12 h-12 rounded-full border-4 border-blue-900 border-t-blue-400 animate-spin"></div>
                  </div>
                  <div>
                    <h3 className="text-blue-400 font-semibold">Running Analysis...</h3>
                    <p className="text-gray-400 text-sm">
                      LangGraph agents are working. This takes ~10-15 seconds.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Results */}
            {result && !loading && (
              <>
                {/* Signal badge */}
                <SignalBadge
                  signal={result.signal}
                  ticker={result.ticker}
                  processingTimeMs={result.processing_time_ms}
                />

                {/* Equity curve chart */}
                <BacktestChart
                  equityCurve={result.equity_curve}
                  startDate={result.backtest_results?.start_date}
                  endDate={result.backtest_results?.end_date}
                  strategyName={result.selected_strategy}
                />

                {/* Metrics table */}
                <MetricsTable
                  backtestResults={result.backtest_results}
                  riskMetrics={result.risk_metrics}
                  marketData={result.market_data}
                />

                {/* AI explanation with citations */}
                <RAGExplainer
                  explanation={result.final_explanation}
                  citations={result.final_citations}
                  strategyRationale={result.strategy_rationale}
                  selectedStrategy={result.selected_strategy}
                />

                {/* Error notice (non-fatal) */}
                {result.error && (
                  <div className="card border border-yellow-800/50 bg-yellow-900/10">
                    <p className="text-yellow-400 text-sm">
                      ⚠️ Partial result: {result.error}
                    </p>
                  </div>
                )}
              </>
            )}

            {/* Empty state */}
            {!result && !loading && !error && (
              <div className="card border-dashed border-gray-700 text-center py-16">
                <div className="text-5xl mb-4">📈</div>
                <h3 className="text-xl font-semibold text-gray-400 mb-2">
                  Ready to Analyze
                </h3>
                <p className="text-gray-500 text-sm max-w-sm mx-auto">
                  Enter a ticker symbol and your question to get an AI-powered
                  trading analysis backed by SEC filings and real backtesting data.
                </p>
                <div className="mt-6 flex flex-wrap justify-center gap-2">
                  {["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"].map((t) => (
                    <span
                      key={t}
                      className="text-xs text-gray-500 bg-gray-800 px-3 py-1 rounded-full"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-16 py-6 text-center text-xs text-gray-600">
        QuantMind — LangGraph + RAG + DSA | Phase 4 MERN Dashboard |{" "}
        <a
          href="https://github.com/MohanpandeyA/quantmind"
          target="_blank"
          rel="noopener noreferrer"
          className="text-gray-500 hover:text-gray-400 underline"
        >
          GitHub
        </a>
      </footer>
    </div>
  );
};

export default App;
