/**
 * API service — calls FastAPI directly via Vite proxy.
 *
 * Vite proxies /api/* → FastAPI (port 8000) with path rewrite:
 *   /api/analyze → /analyze
 *   /api/portfolio → /portfolio
 *   /api/compare → /compare
 *   etc.
 *
 * Express (port 5000) is optional — only needed for MongoDB persistence.
 * When Express is running, analysis results are saved to MongoDB.
 * When Express is not running, analysis still works via FastAPI directly.
 */

import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000, // 2 minutes — analysis can take ~15s
  headers: { "Content-Type": "application/json" },
});

// ---------------------------------------------------------------------------
// Analysis API
// ---------------------------------------------------------------------------

/**
 * Run full trading analysis for a ticker.
 *
 * Triggers the LangGraph workflow:
 * ResearchAgent → RAGAgent → StrategyAgent → BacktestAgent
 * → RiskAgent → ExplainerAgent
 *
 * @param {Object} params
 * @param {string} params.ticker - Stock ticker (e.g., 'AAPL')
 * @param {string} params.query - User's question
 * @param {string} [params.startDate] - Backtest start (YYYY-MM-DD)
 * @param {string} [params.endDate] - Backtest end (YYYY-MM-DD)
 * @returns {Promise<AnalysisResult>}
 *
 * @example
 * const result = await runAnalysis({
 *   ticker: "AAPL",
 *   query: "Should I buy Apple stock?",
 * });
 * console.log(result.signal); // "BUY"
 */
export const runAnalysis = async ({
  ticker,
  query,
  startDate = "2022-01-01",
  endDate = "2024-12-31",
}) => {
  const response = await apiClient.post("/analyze", {
    ticker: ticker.toUpperCase(),
    query,
    start_date: startDate,
    end_date: endDate,
  });
  return response.data;
};

/**
 * Get analysis history from MongoDB.
 *
 * @param {Object} [params]
 * @param {string} [params.ticker] - Filter by ticker
 * @param {number} [params.limit] - Max results (default 20)
 * @returns {Promise<{count: number, analyses: Analysis[]}>}
 */
export const getAnalysisHistory = async ({ ticker, limit = 20 } = {}) => {
  const params = { limit };
  if (ticker) params.ticker = ticker.toUpperCase();
  const response = await apiClient.get("/analysis", { params });
  return response.data;
};

/**
 * Get analyses for a specific ticker.
 *
 * @param {string} ticker - Ticker symbol
 * @returns {Promise<{ticker: string, count: number, analyses: Analysis[]}>}
 */
export const getTickerHistory = async (ticker) => {
  const response = await apiClient.get(`/analysis/${ticker.toUpperCase()}`);
  return response.data;
};

/**
 * Delete an analysis record.
 *
 * @param {string} id - MongoDB document ID
 * @returns {Promise<{message: string, id: string}>}
 */
export const deleteAnalysis = async (id) => {
  const response = await apiClient.delete(`/analysis/${id}`);
  return response.data;
};

/**
 * Check server health.
 *
 * @returns {Promise<{status: string, python_api: string}>}
 */
export const checkHealth = async () => {
  const response = await apiClient.get("/health", {
    baseURL: import.meta.env.VITE_API_URL || "http://localhost:5000",
  });
  return response.data;
};
