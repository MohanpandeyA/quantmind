const fmt = {
  pct: (v) => v != null ? `${(v * 100).toFixed(1)}%` : "--",
  num: (v, d = 2) => v != null ? v.toFixed(d) : "--",
  price: (v) => v != null ? `$${v.toFixed(2)}` : "--",
  int: (v) => v != null ? Math.round(v).toString() : "--",
};

const MetricCard = ({ label, value, sub, color }) => (
  <div className="metric-card flex-1 min-w-0">
    <div className={`metric-value ${color || "text-slate-900"}`}>{value}</div>
    <div className="metric-label">{label}</div>
    {sub && <div className="text-xs text-slate-400 mt-1">{sub}</div>}
  </div>
);

const MetricsTable = ({ backtestResults, riskMetrics, marketData }) => {
  if (!backtestResults && !riskMetrics && !marketData) return null;

  const sharpe = backtestResults?.sharpe_ratio;
  const sharpeColor = sharpe >= 1 ? "text-emerald-600" : sharpe >= 0.5 ? "text-amber-600" : "text-red-600";
  const mdd = backtestResults?.max_drawdown;
  const mddColor = mdd <= 0.1 ? "text-emerald-600" : mdd <= 0.2 ? "text-amber-600" : "text-red-600";
  const winRate = backtestResults?.win_rate;
  const winColor = winRate >= 0.6 ? "text-emerald-600" : winRate >= 0.4 ? "text-amber-600" : "text-red-600";

  return (
    <div className="card">
      <p className="section-title mb-4">Performance Metrics</p>

      <div className="flex gap-3 flex-wrap">
        {sharpe != null && (
          <MetricCard label="Sharpe Ratio" value={fmt.num(sharpe)} sub="Risk-adjusted return" color={sharpeColor} />
        )}
        {backtestResults?.total_return != null && (
          <MetricCard
            label="Total Return"
            value={fmt.pct(backtestResults.total_return)}
            sub={`${backtestResults.start_date || ""} to ${backtestResults.end_date || ""}`}
            color={backtestResults.total_return >= 0 ? "text-emerald-600" : "text-red-600"}
          />
        )}
        {mdd != null && (
          <MetricCard label="Max Drawdown" value={fmt.pct(mdd)} sub="Worst peak-to-trough" color={mddColor} />
        )}
        {winRate != null && (
          <MetricCard label="Win Rate" value={fmt.pct(winRate)} sub={`${fmt.int(backtestResults?.n_trades)} trades`} color={winColor} />
        )}
      </div>

      {riskMetrics && (
        <>
          <div className="divider" />
          <div className="flex gap-3 flex-wrap">
            {riskMetrics.var_95 != null && (
              <MetricCard label="VaR 95%" value={fmt.pct(riskMetrics.var_95)} sub="Daily value at risk" />
            )}
            {riskMetrics.cvar_95 != null && (
              <MetricCard label="CVaR 95%" value={fmt.pct(riskMetrics.cvar_95)} sub="Expected shortfall" />
            )}
            {riskMetrics.risk_level && (
              <MetricCard
                label="Risk Level"
                value={riskMetrics.risk_level}
                sub={`Score: ${fmt.num(riskMetrics.risk_score, 1)}`}
                color={riskMetrics.risk_level === "LOW" ? "text-emerald-600" : riskMetrics.risk_level === "MEDIUM" ? "text-amber-600" : "text-red-600"}
              />
            )}
            {riskMetrics.risk_approved != null && (
              <MetricCard
                label="Risk Check"
                value={riskMetrics.risk_approved ? "Approved" : "Warning"}
                sub={riskMetrics.risk_approved ? "Within limits" : "Exceeded limits"}
                color={riskMetrics.risk_approved ? "text-emerald-600" : "text-amber-600"}
              />
            )}
          </div>
        </>
      )}

      {marketData && (
        <>
          <div className="divider" />
          <div className="flex gap-3 flex-wrap">
            {marketData.current_price != null && (
              <MetricCard
                label="Current Price"
                value={fmt.price(marketData.current_price)}
                sub={marketData.price_change_pct != null ? `${marketData.price_change_pct >= 0 ? "+" : ""}${marketData.price_change_pct?.toFixed(2)}% today` : undefined}
                color={marketData.price_change_pct >= 0 ? "text-emerald-600" : "text-red-600"}
              />
            )}
            {marketData.week_52_high != null && <MetricCard label="52W High" value={fmt.price(marketData.week_52_high)} />}
            {marketData.week_52_low != null && <MetricCard label="52W Low" value={fmt.price(marketData.week_52_low)} />}
            {marketData.volume != null && (
              <MetricCard label="Volume" value={marketData.volume > 1e6 ? `${(marketData.volume / 1e6).toFixed(1)}M` : marketData.volume?.toLocaleString()} />
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default MetricsTable;
