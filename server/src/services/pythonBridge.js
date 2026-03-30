/**
 * Python Bridge — calls the FastAPI backend (Phase 3).
 *
 * The Express server acts as a proxy between the React frontend
 * and the Python FastAPI backend. This separation allows:
 * - MongoDB persistence (save analysis history)
 * - Rate limiting at the Node.js layer
 * - Future auth middleware without touching Python code
 * - CORS handled in one place
 *
 * Architecture:
 *   React → Express (Node.js) → FastAPI (Python) → LangGraph
 */

const axios = require("axios");
const logger = require("../config/logger");

const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://localhost:8000";

// Axios instance with timeout and base URL
const pythonClient = axios.create({
  baseURL: PYTHON_API_URL,
  timeout: 120_000, // 2 minutes — LangGraph analysis can take ~15s
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * Run full trading analysis via FastAPI LangGraph workflow.
 *
 * @param {Object} params - Analysis parameters
 * @param {string} params.ticker - Stock ticker symbol
 * @param {string} params.query - User's natural language question
 * @param {string} params.start_date - Backtest start date (YYYY-MM-DD)
 * @param {string} params.end_date - Backtest end date (YYYY-MM-DD)
 * @returns {Promise<Object>} Analysis result from FastAPI
 * @throws {Error} If FastAPI is unreachable or returns an error
 *
 * @example
 * const result = await runAnalysis({
 *   ticker: "AAPL",
 *   query: "Should I buy Apple stock?",
 *   start_date: "2022-01-01",
 *   end_date: "2024-12-31",
 * });
 * console.log(result.signal); // "BUY"
 */
const runAnalysis = async ({ ticker, query, start_date, end_date }) => {
  logger.info("PythonBridge | calling FastAPI | ticker=%s", ticker);

  try {
    const response = await pythonClient.post("/analyze", {
      ticker,
      query,
      start_date: start_date || "2022-01-01",
      end_date: end_date || "2024-12-31",
    });

    logger.info(
      "PythonBridge | success | ticker=%s | signal=%s | time=%.0fms",
      ticker,
      response.data.signal,
      response.data.processing_time_ms
    );

    return response.data;
  } catch (err) {
    if (err.code === "ECONNREFUSED") {
      const msg =
        "FastAPI backend is not running. " +
        "Start it with: cd backend && uvicorn api.main:app --port 8000";
      logger.error("PythonBridge | connection refused | %s", msg);
      throw new Error(msg);
    }

    if (err.response) {
      const msg = err.response.data?.detail || err.response.statusText;
      logger.error(
        "PythonBridge | FastAPI error | status=%d | %s",
        err.response.status,
        msg
      );
      throw new Error(`FastAPI error: ${msg}`);
    }

    logger.error("PythonBridge | unexpected error | %s", err.message);
    throw err;
  }
};

/**
 * Check if the FastAPI backend is healthy.
 *
 * @returns {Promise<boolean>} True if FastAPI is running
 */
const checkHealth = async () => {
  try {
    const response = await pythonClient.get("/health", { timeout: 3000 });
    return response.status === 200;
  } catch {
    return false;
  }
};

module.exports = { runAnalysis, checkHealth };
