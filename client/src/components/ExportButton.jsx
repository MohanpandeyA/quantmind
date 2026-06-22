import { useState } from "react";
import { exportAnalysisPDF } from "../utils/exportPDF";
import { exportEquityCurveCSV, exportPortfolioCSV, exportWalkForwardCSV } from "../utils/exportCSV";

/**
 * ExportButton — reusable dropdown export button.
 *
 * Props:
 *   type: "analysis" | "portfolio" | "walkforward" | "equity"
 *   data: the data to export (result, positions, windows, equityCurve)
 *   ticker: stock ticker
 *   signal: BUY/SELL/HOLD (for analysis PDF)
 *   strategyName: strategy name (for filenames)
 *   elementId: DOM element ID to capture for PDF
 */
const ExportButton = ({ type, data, ticker, signal, strategyName, elementId, className = "" }) => {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  const handlePDF = async () => {
    setLoading(true);
    setOpen(false);
    await exportAnalysisPDF(elementId || "analysis-result", ticker, signal);
    setLoading(false);
  };

  const handleCSV = () => {
    setOpen(false);
    if (type === "portfolio") {
      exportPortfolioCSV(data);
    } else if (type === "walkforward") {
      exportWalkForwardCSV(data?.windows, ticker, strategyName);
    } else if (type === "equity" || type === "analysis") {
      exportEquityCurveCSV(data?.equity_curve || data, ticker, strategyName);
    }
  };

  const showPDF = type === "analysis";
  const showCSV = true;

  return (
    <div style={{ position: "relative", display: "inline-block" }}>
      <button
        onClick={() => setOpen(!open)}
        disabled={loading}
        style={{
          display: "flex", alignItems: "center", gap: "6px",
          padding: "8px 14px",
          fontSize: "12px", fontWeight: "600",
          color: "#6B7280",
          background: "#F9FAFB",
          border: "1px solid #E5E7EB",
          borderRadius: "8px",
          cursor: "pointer",
          transition: "all 0.15s",
          fontFamily: "inherit",
        }}
      >
        {loading ? (
          <>
            <span style={{ width: "12px", height: "12px", border: "2px solid #E5E7EB", borderTopColor: "#6366F1", borderRadius: "50%", animation: "spin 0.75s linear infinite", display: "inline-block" }}></span>
            Exporting...
          </>
        ) : (
          <>
            <span>⬇</span>
            Export
            <span style={{ fontSize: "10px" }}>▾</span>
          </>
        )}
      </button>

      {open && (
        <>
          <div
            onClick={() => setOpen(false)}
            style={{ position: "fixed", inset: 0, zIndex: 40 }}
          />
          <div style={{
            position: "absolute", top: "calc(100% + 4px)", right: 0,
            background: "#FFFFFF",
            border: "1px solid #E5E7EB",
            borderRadius: "10px",
            boxShadow: "0 8px 24px rgba(0,0,0,0.12)",
            zIndex: 50,
            minWidth: "160px",
            overflow: "hidden",
          }}>
            {showPDF && (
              <button
                onClick={handlePDF}
                style={{
                  display: "flex", alignItems: "center", gap: "8px",
                  width: "100%", padding: "10px 14px",
                  fontSize: "13px", color: "#374151",
                  background: "none", border: "none",
                  cursor: "pointer", textAlign: "left",
                  fontFamily: "inherit",
                  transition: "background 0.1s",
                }}
                onMouseEnter={(e) => e.target.style.background = "#F9FAFB"}
                onMouseLeave={(e) => e.target.style.background = "none"}
              >
                <span>📄</span> Export PDF
              </button>
            )}
            {showCSV && (
              <button
                onClick={handleCSV}
                style={{
                  display: "flex", alignItems: "center", gap: "8px",
                  width: "100%", padding: "10px 14px",
                  fontSize: "13px", color: "#374151",
                  background: "none", border: "none",
                  cursor: "pointer", textAlign: "left",
                  fontFamily: "inherit",
                  transition: "background 0.1s",
                  borderTop: showPDF ? "1px solid #F3F4F6" : "none",
                }}
                onMouseEnter={(e) => e.target.style.background = "#F9FAFB"}
                onMouseLeave={(e) => e.target.style.background = "none"}
              >
                <span>📊</span> Export CSV
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default ExportButton;
