/**
 * BacktestChart — equity curve visualization using Recharts.
 *
 * Shows the portfolio value over the backtest period.
 * A rising curve = strategy was profitable.
 * Drawdowns visible as dips from the peak.
 *
 * Uses Recharts (free, no API key) — a React charting library
 * built on D3.js with a simple declarative API.
 */

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

/**
 * @param {Object} props
 * @param {number[]} props.equityCurve - Array of portfolio values
 * @param {string} [props.startDate] - Backtest start date
 * @param {string} [props.endDate] - Backtest end date
 * @param {string} [props.strategyName] - Strategy name for title
 */
const BacktestChart = ({ equityCurve, startDate, endDate, strategyName }) => {
  if (!equityCurve || equityCurve.length === 0) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-300 mb-4">📊 Equity Curve</h3>
        <div className="h-48 flex items-center justify-center text-gray-500">
          No equity curve data available
        </div>
      </div>
    );
  }

  // Build chart data with approximate dates
  const initialValue = equityCurve[0] || 100_000;
  const data = equityCurve.map((value, i) => ({
    day: i,
    value: Math.round(value),
    return: (((value - initialValue) / initialValue) * 100).toFixed(2),
  }));

  const finalValue = equityCurve[equityCurve.length - 1];
  const totalReturn = ((finalValue - initialValue) / initialValue) * 100;
  const isPositive = totalReturn >= 0;

  // Custom tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload;
      return (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 text-sm">
          <p className="text-gray-400">Day {d.day}</p>
          <p className="text-white font-semibold">
            ${d.value.toLocaleString()}
          </p>
          <p className={d.return >= 0 ? "text-green-400" : "text-red-400"}>
            {d.return >= 0 ? "+" : ""}{d.return}%
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-300">
          📊 Equity Curve
          {strategyName && (
            <span className="text-sm text-gray-500 ml-2 font-normal">
              ({strategyName})
            </span>
          )}
        </h3>
        <div className="text-right">
          <div
            className={`text-xl font-bold ${
              isPositive ? "text-green-400" : "text-red-400"
            }`}
          >
            {isPositive ? "+" : ""}
            {totalReturn.toFixed(1)}%
          </div>
          <div className="text-xs text-gray-500">
            {startDate} → {endDate}
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
          <defs>
            <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="5%"
                stopColor={isPositive ? "#22c55e" : "#ef4444"}
                stopOpacity={0.3}
              />
              <stop
                offset="95%"
                stopColor={isPositive ? "#22c55e" : "#ef4444"}
                stopOpacity={0}
              />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="day"
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            label={{ value: "Trading Days", position: "insideBottom", fill: "#6b7280", fontSize: 11 }}
          />
          <YAxis
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            y={initialValue}
            stroke="#6b7280"
            strokeDasharray="4 4"
            label={{ value: "Start", fill: "#6b7280", fontSize: 10 }}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke={isPositive ? "#22c55e" : "#ef4444"}
            strokeWidth={2}
            fill="url(#equityGradient)"
            dot={false}
            activeDot={{ r: 4, fill: isPositive ? "#22c55e" : "#ef4444" }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default BacktestChart;
