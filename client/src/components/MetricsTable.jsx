/**
 * MetricsTable — displays backtest performance metrics and risk assessment.
 *
 * Shows all key metrics from Phase 1's metrics.py:
 * - Sharpe Ratio (risk-adjusted return)
 * - Max Drawdown (worst loss)
 * - Total Return
 * - Win Rate
 * - VaR 95% (daily loss threshold)
 * - Risk Level (LOW/MEDIUM/HIGH)
 */

/**
 * @param {Object} props
 * @param {Object} props.backtestResults - Backtest metrics from API
 * @param {Object} props.riskMetrics - Risk assessment from API
 * @param {Object} [props.marketData] - Current market data
 */
const MetricsTable = ({ backtestResults, riskMetrics, marketData }) => {
  if (!backtestResults) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-300 mb-4">📈 Performance Metrics</h3>
        <div className="text-gray-500 text-center py-8">No metrics available</div>
      </div>
    );
  }

  const bt = backtestResults;
  const rm = riskMetrics || {};

  // Color helpers
  const returnColor = (v) => (v >= 0 ? "text-green-400" : "text-red-400");
  const sharpeColor = (v) => (v >= 1.0 ? "text-green-400" : v >= 0.5 ? "text-yellow-400" : "text-red-400");
  const ddColor = (v) => (v <= 0.1 ? "text-green-400" : v <= 0.2 ? "text-yellow-400" : "text-red-400");
  const riskLevelColor = (level) => ({
    LOW: "text-green-400 bg-green-900/30 border-green-700",
    MEDIUM: "text-yellow-400 bg-yellow-900/30 border-yellow-700",
    HIGH: "text-red-400 bg-red-900/30 border-red-700",
  }[level] || "text-gray-400");

  const fmt = {
    pct: (v) => v != null ? `${(v * 100).toFixed(1)}%` : "—",
    num: (v) => v != null ? v.toFixed(2) : "—",
    int: (v) => v != null ? v.toLocaleString() : "—",
    price: (v) => v != null ? `$${v.toFixed(2)}` : "—",
  };

  const metrics = [
    {
      label: "Total Return",
      value: fmt.pct(bt.total_return),
      color: returnColor(bt.total_return),
      tooltip: "Total portfolio return over the backtest period",
    },
    {
      label: "Sharpe Ratio",
      value: fmt.num(bt.sharpe_ratio),
      color: sharpeColor(bt.sharpe_ratio),
      tooltip: "Risk-adjusted return. >1.0 = good, >2.0 = excellent",
    },
    {
      label: "Max Drawdown",
      value: fmt.pct(bt.max_drawdown),
      color: ddColor(bt.max_drawdown),
      tooltip: "Worst peak-to-trough loss. Lower is better.",
    },
    {
      label: "Win Rate",
      value: fmt.pct(bt.win_rate),
      color: bt.win_rate >= 0.5 ? "text-green-400" : "text-yellow-400",
      tooltip: "Percentage of profitable trades",
    },
    {
      label: "Sortino Ratio",
      value: fmt.num(bt.sortino_ratio),
      color: sharpeColor(bt.sortino_ratio),
      tooltip: "Like Sharpe but only penalizes downside volatility",
    },
    {
      label: "Profit Factor",
      value: fmt.num(bt.profit_factor),
      color: bt.profit_factor >= 1.5 ? "text-green-400" : "text-yellow-400",
      tooltip: "Gross profit / gross loss. >1.5 = good",
    },
    {
      label: "VaR 95%",
      value: fmt.pct(bt.var_95),
      color: bt.var_95 <= 0.03 ? "text-green-400" : "text-yellow-400",
      tooltip: "Daily loss not exceeded on 95% of days",
    },
    {
      label: "CVaR 95%",
      value: fmt.pct(bt.cvar_95),
      color: "text-gray-300",
      tooltip: "Expected loss on the worst 5% of days",
    },
    {
      label: "Total Trades",
      value: fmt.int(bt.n_trades),
      color: "text-gray-300",
      tooltip: "Number of trades executed in backtest",
    },
    {
      label: "Trading Days",
      value: fmt.int(bt.n_days),
      color: "text-gray-300",
      tooltip: "Total trading days in backtest period",
    },
  ];

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-300">📈 Performance Metrics</h3>
        {rm.risk_level && (
          <span
            className={`text-xs font-semibold px-3 py-1 rounded-full border ${riskLevelColor(rm.risk_level)}`}
          >
            {rm.risk_level} RISK
          </span>
        )}
      </div>

      {/* Strategy info */}
      {bt.strategy_name && (
        <div className="mb-4 text-sm text-gray-400 bg-gray-800/50 rounded-lg px-3 py-2">
          Strategy: <span className="text-blue-400 font-mono">{bt.strategy_name}</span>
          <span className="mx-2 text-gray-600">|</span>
          {bt.start_date} → {bt.end_date}
        </div>
      )}

      {/* Metrics grid */}
      <div className="grid grid-cols-2 gap-3">
        {metrics.map(({ label, value, color, tooltip }) => (
          <div
            key={label}
            className="bg-gray-800/50 rounded-lg p-3 hover:bg-gray-800 transition-colors"
            title={tooltip}
          >
            <div className={`metric-value ${color}`}>{value}</div>
            <div className="metric-label">{label}</div>
          </div>
        ))}
      </div>

      {/* Market data strip */}
      {marketData && (
        <div className="mt-4 pt-4 border-t border-gray-800 grid grid-cols-3 gap-3 text-sm">
          <div>
            <div className="text-gray-400 text-xs">Current Price</div>
            <div className="text-white font-semibold">{fmt.price(marketData.current_price)}</div>
          </div>
          <div>
            <div className="text-gray-400 text-xs">Today's Change</div>
            <div className={`font-semibold ${returnColor(marketData.price_change_pct)}`}>
              {marketData.price_change_pct >= 0 ? "+" : ""}
              {marketData.price_change_pct?.toFixed(2)}%
            </div>
          </div>
          <div>
            <div className="text-gray-400 text-xs">52W Range</div>
            <div className="text-gray-300 text-xs">
              {fmt.price(marketData.week_52_low)} — {fmt.price(marketData.week_52_high)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MetricsTable;
