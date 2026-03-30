/**
 * Global error handler middleware for Express.
 * Catches all errors thrown in route handlers and returns
 * a consistent JSON error response.
 *
 * Must be registered LAST in the middleware chain.
 */

const logger = require("../config/logger");

/**
 * Express error handler middleware.
 *
 * @param {Error} err - The error object
 * @param {import('express').Request} req - Express request
 * @param {import('express').Response} res - Express response
 * @param {import('express').NextFunction} next - Next middleware
 */
const errorHandler = (err, req, res, next) => {
  const statusCode = err.statusCode || err.status || 500;
  const message = err.message || "Internal Server Error";

  logger.error("Error | %s %s | status=%d | %s", req.method, req.path, statusCode, message);

  res.status(statusCode).json({
    error: message,
    path: req.path,
    timestamp: new Date().toISOString(),
    ...(process.env.NODE_ENV === "development" && { stack: err.stack }),
  });
};

module.exports = errorHandler;
