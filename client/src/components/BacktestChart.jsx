import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const val = payload[0]?.value;
  return (
    <div className="bg-white border border-slate-100 rounded-xl shadow-card-lg px-3 py-2">
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      <p className="text-sm font-semibold text-slate-900">
        ${val?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
      </p>
    </div>
  );
};

const BacktestChart = ({ equityCurve, startDate, endDate, strategyName }) => {
  if (!equityCurve?.length) return null;

  const data = equityCurve.map((point, i) => ({
    date: point.date || `Day ${i + 1}`,
    value: typeof point === "number" ? point : point.value ?? point,
  }));

  const startVal = data[0]?.value ?? 100000;
  const endVal = data[data.length - 1]?.value ?? startVal;
  const totalReturn = ((endVal - startVal) / startVal) * 100;
  const isPositive = totalReturn >= 0;

  const tickCount = Math.min(6, data.length);
  const step = Math.floor(data.length / tickCount);
  const ticks = data.filter((_, i) => i % step === 0).map((d) => d.date);

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="section-title">Equity Curve</p>
          {strategyName && (
            <p className="text-xs text-slate-400 mt-0.5 capitalize">{strategyName.replace(/_/g, " ")} strategy</p>
          )}
        </div>
        <div className="text-right">
          <div className={`text-lg font-bold tabular-nums ${isPositive ? "text-emerald-600" : "text-red-600"}`}>
            {isPositive ? "+" : ""}{totalReturn.toFixed(1)}%
          </div>
          <div className="text-xs text-slate-400">
            {startDate} to {endDate}
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={data} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={isPositive ? "#4f46e5" : "#dc2626"} stopOpacity={0.12} />
              <stop offset="95%" stopColor={isPositive ? "#4f46e5" : "#dc2626"} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
          <XAxis
            dataKey="date"
            ticks={ticks}
            tick={{ fontSize: 10, fill: "#94a3b8" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "#94a3b8" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
            width={48}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={startVal} stroke="#e2e8f0" strokeDasharray="4 4" />
          <Area
            type="monotone"
            dataKey="value"
            stroke={isPositive ? "#4f46e5" : "#dc2626"}
            strokeWidth={2}
            fill="url(#equityGrad)"
            dot={false}
            activeDot={{ r: 4, fill: isPositive ? "#4f46e5" : "#dc2626", strokeWidth: 0 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default BacktestChart;
