export const exportCSV = (data, filename = "export.csv") => {
  if (!data?.length) return;
  const headers = Object.keys(data[0]);
  const rows = data.map((row) =>
    headers.map((h) => {
      const val = row[h];
      if (val === null || val === undefined) return "";
      const str = String(val);
      return str.includes(",") || str.includes('"') || str.includes("\n")
        ? `"${str.replace(/"/g, '""')}"`
        : str;
    }).join(",")
  );
  const csv = [headers.join(","), ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
};

export const exportEquityCurveCSV = (equityCurve, ticker, strategyName) => {
  if (!equityCurve?.length) return;
  const data = equityCurve.map((point, i) => ({
    index: i + 1,
    date: point.date || `Day ${i + 1}`,
    portfolio_value: typeof point === "number" ? point : (point.value ?? point),
  }));
  exportCSV(data, `${ticker}_${strategyName || "backtest"}_equity_curve.csv`);
};

export const exportPortfolioCSV = (positions) => {
  if (!positions?.length) return;
  const data = positions.map((p) => ({
    ticker: p.ticker,
    shares: p.shares,
    entry_price: p.entry_price,
    current_price: p.current_price,
    cost_basis: p.cost_basis,
    current_value: p.current_value,
    unrealized_pnl: p.unrealized_pnl,
    unrealized_pnl_pct: p.unrealized_pnl_pct ? `${(p.unrealized_pnl_pct * 100).toFixed(2)}%` : "",
    entry_date: p.entry_date || "",
    notes: p.notes || "",
  }));
  exportCSV(data, `portfolio_${new Date().toISOString().split("T")[0]}.csv`);
};

export const exportWalkForwardCSV = (windows, ticker, strategy) => {
  if (!windows?.length) return;
  const data = windows.map((w) => ({
    window: w.window,
    train_start: w.train_start,
    train_end: w.train_end,
    test_start: w.test_start,
    test_end: w.test_end,
    train_sharpe: w.train_sharpe?.toFixed(3),
    test_sharpe: w.test_sharpe?.toFixed(3),
    best_params: JSON.stringify(w.best_params || {}),
  }));
  exportCSV(data, `${ticker}_${strategy}_walk_forward.csv`);
};
