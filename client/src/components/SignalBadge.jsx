/**
 * SignalBadge — displays the BUY/SELL/HOLD trading signal.
 *
 * Color coding:
 *   BUY  → green (positive, go long)
 *   SELL → red   (negative, exit/short)
 *   HOLD → yellow (neutral, wait)
 */

/**
 * @param {Object} props
 * @param {string} props.signal - "BUY", "SELL", or "HOLD"
 * @param {string} [props.ticker] - Ticker symbol for display
 * @param {number} [props.processingTimeMs] - Analysis time in ms
 */
const SignalBadge = ({ signal, ticker, processingTimeMs }) => {
  const config = {
    BUY: {
      bg: "bg-green-900/40",
      border: "border-green-500",
      text: "text-green-400",
      icon: "📈",
      label: "BUY",
      description: "Positive momentum — consider entering a long position",
    },
    SELL: {
      bg: "bg-red-900/40",
      border: "border-red-500",
      text: "text-red-400",
      icon: "📉",
      label: "SELL",
      description: "Negative momentum — consider exiting or avoiding",
    },
    HOLD: {
      bg: "bg-yellow-900/40",
      border: "border-yellow-500",
      text: "text-yellow-400",
      icon: "⏸️",
      label: "HOLD",
      description: "Neutral — wait for a clearer signal",
    },
  };

  const c = config[signal] || config.HOLD;

  return (
    <div className={`card border-2 ${c.border} ${c.bg}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <span className="text-5xl">{c.icon}</span>
          <div>
            <div className="flex items-center gap-3">
              {ticker && (
                <span className="text-gray-400 text-lg font-mono">{ticker}</span>
              )}
              <span className={`text-4xl font-black ${c.text}`}>{c.label}</span>
            </div>
            <p className="text-gray-400 text-sm mt-1">{c.description}</p>
          </div>
        </div>

        {processingTimeMs && (
          <div className="text-right">
            <div className="text-xs text-gray-500">Analysis time</div>
            <div className="text-sm text-gray-400 font-mono">
              {(processingTimeMs / 1000).toFixed(1)}s
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SignalBadge;
