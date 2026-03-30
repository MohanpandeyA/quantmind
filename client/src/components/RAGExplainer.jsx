/**
 * RAGExplainer — displays the AI-generated explanation with source citations.
 *
 * This is the most important component — it shows WHY the signal was generated,
 * backed by real financial documents (SEC filings, news articles).
 *
 * Features:
 * - Formatted explanation text with signal highlighted
 * - Numbered source citations (from Phase 2 RAG pipeline)
 * - Strategy rationale from StrategyAgent
 * - Expandable/collapsible sections
 */

import { useState } from "react";

/**
 * @param {Object} props
 * @param {string} props.explanation - LLM-generated explanation text
 * @param {string[]} props.citations - Source citation strings
 * @param {string} [props.strategyRationale] - Why this strategy was selected
 * @param {string} [props.selectedStrategy] - Strategy name
 */
const RAGExplainer = ({
  explanation,
  citations,
  strategyRationale,
  selectedStrategy,
}) => {
  const [showCitations, setShowCitations] = useState(true);
  const [showStrategy, setShowStrategy] = useState(false);

  if (!explanation) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-300 mb-4">
          🤖 AI Analysis
        </h3>
        <div className="text-gray-500 text-center py-8">
          No explanation available
        </div>
      </div>
    );
  }

  // Parse signal line from explanation
  const lines = explanation.split("\n").filter((l) => l.trim());
  const signalLine = lines.find((l) => l.startsWith("SIGNAL:"));
  const confidenceLine = lines.find((l) => l.startsWith("CONFIDENCE:"));
  const bodyLines = lines.filter(
    (l) => !l.startsWith("SIGNAL:") && !l.startsWith("CONFIDENCE:") && !l.startsWith("SOURCES:")
  );

  const signalColors = {
    BUY: "text-green-400",
    SELL: "text-red-400",
    HOLD: "text-yellow-400",
  };

  const getSignalColor = (line) => {
    if (line?.includes("BUY")) return signalColors.BUY;
    if (line?.includes("SELL")) return signalColors.SELL;
    return signalColors.HOLD;
  };

  return (
    <div className="card space-y-4">
      <h3 className="text-lg font-semibold text-gray-300 flex items-center gap-2">
        <span>🤖</span>
        AI Analysis
        {selectedStrategy && (
          <span className="text-xs text-blue-400 bg-blue-900/30 px-2 py-0.5 rounded-full font-normal">
            {selectedStrategy}
          </span>
        )}
      </h3>

      {/* Signal + Confidence header */}
      {(signalLine || confidenceLine) && (
        <div className="flex items-center gap-4 bg-gray-800/50 rounded-lg px-4 py-3">
          {signalLine && (
            <span className={`font-bold text-lg ${getSignalColor(signalLine)}`}>
              {signalLine}
            </span>
          )}
          {confidenceLine && (
            <span className="text-gray-400 text-sm">{confidenceLine}</span>
          )}
        </div>
      )}

      {/* Main explanation body */}
      <div className="prose prose-invert prose-sm max-w-none">
        {bodyLines.map((line, i) => (
          <p key={i} className="text-gray-300 leading-relaxed mb-2">
            {line}
          </p>
        ))}
      </div>

      {/* Strategy rationale (collapsible) */}
      {strategyRationale && (
        <div className="border border-gray-800 rounded-lg overflow-hidden">
          <button
            onClick={() => setShowStrategy(!showStrategy)}
            className="w-full flex items-center justify-between px-4 py-3
                       text-sm text-gray-400 hover:text-gray-300 hover:bg-gray-800/50
                       transition-colors"
          >
            <span className="flex items-center gap-2">
              <span>⚙️</span>
              Strategy Selection Rationale
            </span>
            <span>{showStrategy ? "▼" : "▶"}</span>
          </button>
          {showStrategy && (
            <div className="px-4 py-3 bg-gray-800/30 text-sm text-gray-400 border-t border-gray-800">
              {strategyRationale}
            </div>
          )}
        </div>
      )}

      {/* Source citations (collapsible) */}
      {citations && citations.length > 0 && (
        <div className="border border-gray-800 rounded-lg overflow-hidden">
          <button
            onClick={() => setShowCitations(!showCitations)}
            className="w-full flex items-center justify-between px-4 py-3
                       text-sm text-gray-400 hover:text-gray-300 hover:bg-gray-800/50
                       transition-colors"
          >
            <span className="flex items-center gap-2">
              <span>📚</span>
              Sources ({citations.length})
            </span>
            <span>{showCitations ? "▼" : "▶"}</span>
          </button>
          {showCitations && (
            <div className="px-4 py-3 bg-gray-800/30 border-t border-gray-800 space-y-2">
              {citations.map((citation, i) => (
                <div key={i} className="flex gap-2 text-sm">
                  <span className="text-blue-400 font-mono shrink-0">[{i + 1}]</span>
                  <span className="text-gray-400 break-all">{citation}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* No Groq key notice */}
      {explanation.includes("GROQ_API_KEY") && (
        <div className="bg-yellow-900/20 border border-yellow-800 rounded-lg px-4 py-3 text-sm text-yellow-400">
          💡 Add <code className="bg-yellow-900/40 px-1 rounded">GROQ_API_KEY</code> to{" "}
          <code className="bg-yellow-900/40 px-1 rounded">backend/.env</code> for
          AI-powered explanations (free at{" "}
          <a
            href="https://console.groq.com"
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:text-yellow-300"
          >
            console.groq.com
          </a>
          )
        </div>
      )}
    </div>
  );
};

export default RAGExplainer;
