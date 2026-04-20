/**
 * RAGExplainer — displays the AI-generated explanation with source citations
 * and FinBERT sentiment analysis.
 *
 * Features:
 * - Formatted explanation text with signal highlighted
 * - FinBERT sentiment score + top positive/negative signals (NEW)
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
 * @param {number} [props.sentimentScore] - FinBERT score -1.0 to +1.0
 * @param {string} [props.sentimentLabel] - "BULLISH" / "BEARISH" / "NEUTRAL"
 * @param {number} [props.sentimentConfidence] - 0.0 to 1.0
 * @param {Object} [props.sentimentDetails] - top positive/negative sentences
 */
const RAGExplainer = ({
  explanation,
  citations,
  strategyRationale,
  selectedStrategy,
  sentimentScore,
  sentimentLabel,
  sentimentConfidence,
  sentimentDetails,
}) => {
  const [showCitations, setShowCitations] = useState(true);
  const [showStrategy, setShowStrategy] = useState(false);
  const [showSentiment, setShowSentiment] = useState(true);

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

  // Sentiment display helpers
  const hasSentiment = sentimentLabel != null && sentimentScore != null;
  const sentimentColor = {
    BULLISH: "text-green-400",
    BEARISH: "text-red-400",
    NEUTRAL: "text-yellow-400",
  }[sentimentLabel] || "text-gray-400";

  const sentimentBg = {
    BULLISH: "border-green-800 bg-green-900/10",
    BEARISH: "border-red-800 bg-red-900/10",
    NEUTRAL: "border-yellow-800 bg-yellow-900/10",
  }[sentimentLabel] || "border-gray-800 bg-gray-800/20";

  const sentimentEmoji = {
    BULLISH: "🟢",
    BEARISH: "🔴",
    NEUTRAL: "🟡",
  }[sentimentLabel] || "⚪";

  const topPositive = sentimentDetails?.top_positive || [];
  const topNegative = sentimentDetails?.top_negative || [];
  const totalSentences = sentimentDetails?.total_sentences || 0;

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

      {/* FinBERT Sentiment Section */}
      {hasSentiment && (
        <div className={`border rounded-lg overflow-hidden ${sentimentBg}`}>
          <button
            onClick={() => setShowSentiment(!showSentiment)}
            className="w-full flex items-center justify-between px-4 py-3
                       text-sm hover:bg-gray-800/30 transition-colors"
          >
            <span className="flex items-center gap-2">
              <span>🧠</span>
              <span className="text-gray-300 font-medium">FinBERT Sentiment</span>
              <span className={`font-bold ${sentimentColor}`}>
                {sentimentEmoji} {sentimentLabel}
              </span>
              <span className="text-gray-500 font-mono text-xs">
                ({sentimentScore >= 0 ? "+" : ""}{sentimentScore?.toFixed(3)})
              </span>
              {sentimentConfidence > 0 && (
                <span className="text-gray-600 text-xs">
                  {(sentimentConfidence * 100).toFixed(0)}% conf
                </span>
              )}
            </span>
            <span className="text-gray-500">{showSentiment ? "▼" : "▶"}</span>
          </button>

          {showSentiment && (
            <div className="px-4 py-3 border-t border-gray-800/50 space-y-3">
              {/* Score bar */}
              <div className="flex items-center gap-3">
                <span className="text-xs text-red-400 w-12 text-right">Bearish</span>
                <div className="flex-1 bg-gray-800 rounded-full h-2 relative">
                  {/* Center line */}
                  <div className="absolute top-0 left-1/2 w-px h-full bg-gray-600" />
                  {/* Score indicator */}
                  <div
                    className={`absolute top-0 h-full rounded-full transition-all ${
                      sentimentScore >= 0 ? "bg-green-500" : "bg-red-500"
                    }`}
                    style={{
                      left: sentimentScore >= 0 ? "50%" : `${(sentimentScore + 1) / 2 * 100}%`,
                      width: `${Math.abs(sentimentScore) / 2 * 100}%`,
                    }}
                  />
                </div>
                <span className="text-xs text-green-400 w-12">Bullish</span>
              </div>

              {/* Top signals */}
              <div className="grid grid-cols-2 gap-3">
                {topPositive.length > 0 && (
                  <div>
                    <div className="text-xs text-green-400 font-medium mb-1">
                      🟢 Positive signals
                    </div>
                    {topPositive.slice(0, 2).map((item, i) => (
                      <div key={i} className="text-xs text-gray-400 mb-1 flex gap-1">
                        <span className="text-green-500 font-mono shrink-0">
                          {item.score >= 0 ? "+" : ""}{item.score?.toFixed(2)}
                        </span>
                        <span className="line-clamp-2">{item.text}</span>
                      </div>
                    ))}
                  </div>
                )}
                {topNegative.length > 0 && (
                  <div>
                    <div className="text-xs text-red-400 font-medium mb-1">
                      🔴 Negative signals
                    </div>
                    {topNegative.slice(0, 2).map((item, i) => (
                      <div key={i} className="text-xs text-gray-400 mb-1 flex gap-1">
                        <span className="text-red-500 font-mono shrink-0">
                          {item.score >= 0 ? "+" : ""}{item.score?.toFixed(2)}
                        </span>
                        <span className="line-clamp-2">{item.text}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {totalSentences > 0 && (
                <div className="text-xs text-gray-600">
                  Analyzed {totalSentences} sentences from SEC filings, news
                  {sentimentDetails?.reddit_sentences > 0
                    ? ` + ${sentimentDetails.reddit_sentences} social posts`
                    : ""}
                  {" "}using ProsusAI/finbert
                </div>
              )}
            </div>
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
