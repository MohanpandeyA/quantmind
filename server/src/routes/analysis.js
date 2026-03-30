/**
 * Analysis routes — Express router for trading analysis endpoints.
 *
 * Routes:
 *   POST /api/analysis          → Run full analysis (calls FastAPI)
 *   GET  /api/analysis          → Get analysis history (from MongoDB)
 *   GET  /api/analysis/:ticker  → Get analyses for a specific ticker
 *   DELETE /api/analysis/:id    → Delete an analysis record
 */

const express = require("express");
const { body, param, validationResult } = require("express-validator");
const Analysis = require("../models/Analysis");
const { runAnalysis } = require("../services/pythonBridge");
const logger = require("../config/logger");

const router = express.Router();

// ---------------------------------------------------------------------------
// Validation middleware
// ---------------------------------------------------------------------------

const validateAnalysisRequest = [
  body("ticker")
    .trim()
    .notEmpty()
    .withMessage("ticker is required")
    .isLength({ max: 20 })
    .withMessage("ticker must be <= 20 characters"),
  body("query")
    .trim()
    .notEmpty()
    .withMessage("query is required")
    .isLength({ min: 5, max: 500 })
    .withMessage("query must be 5-500 characters"),
  body("start_date")
    .optional()
    .matches(/^\d{4}-\d{2}-\d{2}$/)
    .withMessage("start_date must be YYYY-MM-DD"),
  body("end_date")
    .optional()
    .matches(/^\d{4}-\d{2}-\d{2}$/)
    .withMessage("end_date must be YYYY-MM-DD"),
];

const handleValidationErrors = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  next();
};

// ---------------------------------------------------------------------------
// POST /api/analysis — Run full analysis
// ---------------------------------------------------------------------------

/**
 * @route   POST /api/analysis
 * @desc    Run full LangGraph trading analysis for a ticker
 * @access  Public
 *
 * @body    { ticker, query, start_date?, end_date? }
 * @returns { signal, final_explanation, backtest_results, ... }
 */
router.post(
  "/",
  validateAnalysisRequest,
  handleValidationErrors,
  async (req, res, next) => {
    const { ticker, query, start_date, end_date } = req.body;
    const tickerUpper = ticker.toUpperCase();

    logger.info("POST /api/analysis | ticker=%s | query=%s", tickerUpper, query.slice(0, 50));

    try {
      // Call FastAPI LangGraph workflow
      const result = await runAnalysis({
        ticker: tickerUpper,
        query,
        start_date: start_date || "2022-01-01",
        end_date: end_date || "2024-12-31",
      });

      // Save to MongoDB (non-blocking — don't fail if DB is down)
      saveAnalysis(result, tickerUpper, query, start_date, end_date).catch(
        (err) => logger.warn("Failed to save analysis to MongoDB: %s", err.message)
      );

      res.json(result);
    } catch (err) {
      next(err);
    }
  }
);

// ---------------------------------------------------------------------------
// GET /api/analysis — Get analysis history
// ---------------------------------------------------------------------------

/**
 * @route   GET /api/analysis
 * @desc    Get recent analysis history from MongoDB
 * @access  Public
 * @query   limit (default 20), ticker (optional filter)
 */
router.get("/", async (req, res, next) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 20, 100);
    const filter = req.query.ticker
      ? { ticker: req.query.ticker.toUpperCase() }
      : {};

    const analyses = await Analysis.find(filter)
      .select("-equityCurve -finalExplanation") // Exclude large fields from list
      .sort({ createdAt: -1 })
      .limit(limit)
      .lean();

    res.json({ count: analyses.length, analyses });
  } catch (err) {
    next(err);
  }
});

// ---------------------------------------------------------------------------
// GET /api/analysis/:ticker — Get analyses for a ticker
// ---------------------------------------------------------------------------

/**
 * @route   GET /api/analysis/:ticker
 * @desc    Get all analyses for a specific ticker
 * @access  Public
 */
router.get(
  "/:ticker",
  [param("ticker").trim().notEmpty().isLength({ max: 20 })],
  handleValidationErrors,
  async (req, res, next) => {
    try {
      const ticker = req.params.ticker.toUpperCase();
      const analyses = await Analysis.find({ ticker })
        .sort({ createdAt: -1 })
        .limit(10)
        .lean();

      res.json({ ticker, count: analyses.length, analyses });
    } catch (err) {
      next(err);
    }
  }
);

// ---------------------------------------------------------------------------
// DELETE /api/analysis/:id — Delete an analysis
// ---------------------------------------------------------------------------

/**
 * @route   DELETE /api/analysis/:id
 * @desc    Delete an analysis record by MongoDB ID
 * @access  Public
 */
router.delete("/:id", async (req, res, next) => {
  try {
    const deleted = await Analysis.findByIdAndDelete(req.params.id);
    if (!deleted) {
      return res.status(404).json({ error: "Analysis not found" });
    }
    res.json({ message: "Analysis deleted", id: req.params.id });
  } catch (err) {
    next(err);
  }
});

// ---------------------------------------------------------------------------
// Helper: save analysis to MongoDB
// ---------------------------------------------------------------------------

/**
 * Save analysis result to MongoDB.
 * Called asynchronously — does not block the API response.
 *
 * @param {Object} result - FastAPI analysis result
 * @param {string} ticker - Ticker symbol
 * @param {string} query - User query
 * @param {string} startDate - Backtest start date
 * @param {string} endDate - Backtest end date
 */
async function saveAnalysis(result, ticker, query, startDate, endDate) {
  const bt = result.backtest_results || {};
  const rm = result.risk_metrics || {};

  const analysis = new Analysis({
    ticker,
    query,
    signal: result.signal || "HOLD",
    finalExplanation: result.final_explanation || "",
    finalCitations: result.final_citations || [],
    selectedStrategy: result.selected_strategy || "",
    strategyRationale: result.strategy_rationale || "",
    backtestResults: bt.strategy_name
      ? {
          strategyName: bt.strategy_name,
          totalReturn: bt.total_return,
          annualizedReturn: bt.annualized_return,
          sharpeRatio: bt.sharpe_ratio,
          sortinoRatio: bt.sortino_ratio,
          maxDrawdown: bt.max_drawdown,
          calmarRatio: bt.calmar_ratio,
          var95: bt.var_95,
          cvar95: bt.cvar_95,
          winRate: bt.win_rate,
          profitFactor: bt.profit_factor,
          nTrades: bt.n_trades,
          nDays: bt.n_days,
          startDate: bt.start_date,
          endDate: bt.end_date,
        }
      : undefined,
    equityCurve: (result.equity_curve || []).slice(0, 252),
    riskMetrics: rm.risk_level
      ? {
          sharpeRatio: rm.sharpe_ratio,
          maxDrawdown: rm.max_drawdown,
          var95: rm.var_95,
          riskScore: rm.risk_score,
          riskLevel: rm.risk_level,
          riskApproved: rm.risk_approved,
          rejectionReason: rm.rejection_reason || "",
        }
      : undefined,
    processingTimeMs: result.processing_time_ms,
    error: result.error || "",
    startDate: startDate || "2022-01-01",
    endDate: endDate || "2024-12-31",
  });

  await analysis.save();
  logger.info("Analysis saved to MongoDB | ticker=%s | id=%s", ticker, analysis._id);
}

module.exports = router;
