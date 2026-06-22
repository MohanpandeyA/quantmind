const SENTIMENT_CONFIG = {
  BULLISH: { color: "text-emerald-600", bg: "bg-emerald-50", border: "border-emerald-100", dot: "bg-emerald-500" },
  BEARISH: { color: "text-red-600", bg: "bg-red-50", border: "border-red-100", dot: "bg-red-500" },
  NEUTRAL: { color: "text-amber-600", bg: "bg-amber-50", border: "border-amber-100", dot: "bg-amber-500" },
};

const RAGExplainer = ({
  explanation, citations, strategyRationale, selectedStrategy,
  sentimentScore, sentimentLabel, sentimentConfidence, sentimentDetails,
}) => {
  if (!explanation && !sentimentLabel) return null;

  const sentCfg = SENTIMENT_CONFIG[sentimentLabel] || SENTIMENT_CONFIG.NEUTRAL;

  const lines = explanation
    ? explanation.split("\n").filter((l) => l.trim())
    : [];

  return (
    <div className="space-y-4">

      {/* Sentiment card */}
      {sentimentLabel && (
        <div className={`card border ${sentCfg.border} ${sentCfg.bg}`}>
          <div className="flex items-center justify-between mb-3">
            <p className="section-title">FinBERT Sentiment</p>
            <div className={`badge ${sentCfg.bg} ${sentCfg.color} border ${sentCfg.border}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${sentCfg.dot}`}></span>
              {sentimentLabel}
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div>
              <div className={`text-2xl font-bold tabular-nums ${sentCfg.color}`}>
                {sentimentScore != null ? `${sentimentScore >= 0 ? "+" : ""}${sentimentScore.toFixed(3)}` : "--"}
              </div>
              <div className="text-xs text-slate-400 mt-0.5">
                Confidence: {sentimentConfidence != null ? `${(sentimentConfidence * 100).toFixed(0)}%` : "--"}
              </div>
            </div>

            {/* Score bar */}
            <div className="flex-1">
              <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${sentCfg.dot}`}
                  style={{ width: `${Math.min(100, Math.abs((sentimentScore || 0)) * 100)}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-slate-400 mt-1">
                <span>Bearish</span>
                <span>Neutral</span>
                <span>Bullish</span>
              </div>
            </div>
          </div>

          {sentimentDetails?.length > 0 && (
            <div className="mt-3 space-y-1.5">
              {sentimentDetails.slice(0, 2).map((d, i) => (
                <div key={i} className="text-xs text-slate-500 bg-white/60 rounded-lg px-3 py-2 border border-white">
                  "{d.text?.slice(0, 120)}{d.text?.length > 120 ? "..." : ""}"
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Strategy rationale */}
      {strategyRationale && (
        <div className="card">
          <p className="section-title mb-2">Strategy Selection</p>
          <div className="flex items-start gap-2">
            {selectedStrategy && (
              <span className="badge badge-indigo capitalize flex-shrink-0">
                {selectedStrategy.replace(/_/g, " ")}
              </span>
            )}
            <p className="text-sm text-slate-600 leading-relaxed">{strategyRationale}</p>
          </div>
        </div>
      )}

      {/* AI explanation */}
      {explanation && (
        <div className="card">
          <p className="section-title mb-3">AI Analysis</p>
          <div className="space-y-2">
            {lines.map((line, i) => {
              if (line.startsWith("SIGNAL:") || line.startsWith("CONFIDENCE:")) {
                return (
                  <div key={i} className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-slate-400 uppercase w-24 flex-shrink-0">
                      {line.split(":")[0]}
                    </span>
                    <span className="text-sm font-semibold text-slate-800">
                      {line.split(":").slice(1).join(":").trim()}
                    </span>
                  </div>
                );
              }
              return (
                <p key={i} className="text-sm text-slate-600 leading-relaxed">
                  {line}
                </p>
              );
            })}
          </div>

          {citations?.length > 0 && (
            <div className="mt-4 pt-4 border-t border-slate-50">
              <p className="text-xs text-slate-400 mb-2">Sources</p>
              <div className="flex flex-wrap gap-1.5">
                {citations.map((c, i) => (
                  <span key={i} className="badge badge-slate text-xs">
                    {c}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default RAGExplainer;
