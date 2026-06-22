const SIGNAL_CONFIG = {
  BUY: {
    border: "border-l-emerald-500",
    bg: "bg-emerald-50",
    text: "text-emerald-700",
    badge: "bg-emerald-500",
    icon: "↑",
    label: "BUY",
  },
  SELL: {
    border: "border-l-red-500",
    bg: "bg-red-50",
    text: "text-red-700",
    badge: "bg-red-500",
    icon: "↓",
    label: "SELL",
  },
  HOLD: {
    border: "border-l-amber-500",
    bg: "bg-amber-50",
    text: "text-amber-700",
    badge: "bg-amber-500",
    icon: "→",
    label: "HOLD",
  },
};

const SignalBadge = ({ signal, ticker, processingTimeMs }) => {
  const cfg = SIGNAL_CONFIG[signal] || SIGNAL_CONFIG.HOLD;

  return (
    <div className={`card border-l-4 ${cfg.border} ${cfg.bg} !p-5`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className={`w-12 h-12 ${cfg.badge} rounded-xl flex items-center justify-center shadow-sm`}>
            <span className="text-white text-xl font-bold">{cfg.icon}</span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className={`text-3xl font-black tracking-tight ${cfg.text}`}>
                {cfg.label}
              </span>
              {ticker && (
                <span className="text-slate-400 text-lg font-mono font-semibold">
                  {ticker}
                </span>
              )}
            </div>
            <p className="text-slate-500 text-sm mt-0.5">
              AI-powered signal based on 7-agent analysis
            </p>
          </div>
        </div>

        {processingTimeMs && (
          <div className="text-right hidden sm:block">
            <div className="text-xs text-slate-400">Processed in</div>
            <div className="text-sm font-semibold text-slate-600 tabular-nums">
              {(processingTimeMs / 1000).toFixed(1)}s
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SignalBadge;
