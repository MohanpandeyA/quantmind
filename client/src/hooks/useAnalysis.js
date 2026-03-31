/**
 * useAnalysis — custom React hook for running trading analysis.
 *
 * Manages the full lifecycle of an analysis request:
 * - Loading state (spinner while LangGraph runs)
 * - Error state (display error message)
 * - Result state (display analysis results)
 *
 * @example
 * const { result, loading, error, analyze } = useAnalysis();
 *
 * // Trigger analysis
 * await analyze({ ticker: "AAPL", query: "Should I buy?" });
 *
 * // Use result
 * if (result) {
 *   console.log(result.signal); // "BUY"
 * }
 */

import { useState, useCallback } from "react";
import { runAnalysis } from "../services/api";

/**
 * @typedef {Object} UseAnalysisReturn
 * @property {Object|null} result - Analysis result from the API
 * @property {boolean} loading - True while analysis is running
 * @property {string|null} error - Error message if analysis failed
 * @property {Function} analyze - Function to trigger analysis
 * @property {Function} reset - Function to clear result and error
 */

/**
 * Custom hook for running trading analysis.
 *
 * @returns {UseAnalysisReturn}
 */
const useAnalysis = () => {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Run analysis for a ticker.
   *
   * @param {Object} params
   * @param {string} params.ticker - Stock ticker
   * @param {string} params.query - User's question
   * @param {string} [params.startDate] - Backtest start date
   * @param {string} [params.endDate] - Backtest end date
   */
  const analyze = useCallback(async ({ ticker, query, startDate, endDate }) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await runAnalysis({ ticker, query, startDate, endDate });
      setResult(data);
    } catch (err) {
      // Handle Pydantic 422 validation errors (array of error objects)
      const detail = err.response?.data?.detail;
      let message;

      if (Array.isArray(detail)) {
        // Pydantic validation error: [{loc: [...], msg: "...", type: "..."}]
        message = detail.map((e) => e.msg).join(". ");
      } else if (typeof detail === "string") {
        message = detail;
      } else {
        message =
          err.response?.data?.error ||
          err.message ||
          "Analysis failed. Please try again.";
      }

      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Reset result and error state.
   */
  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, loading, error, analyze, reset };
};

export default useAnalysis;
