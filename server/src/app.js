/**
 * QuantMind Express Server — MERN backend layer.
 *
 * Acts as a proxy between the React frontend and the Python FastAPI backend.
 * Also handles MongoDB persistence for analysis history.
 *
 * Architecture:
 *   React (port 5173) → Express (port 5000) → FastAPI (port 8000)
 *
 * To run:
 *   cd quantmind/server
 *   npm install
 *   npm run dev
 */

require("dotenv").config();

const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const morgan = require("morgan");
const rateLimit = require("express-rate-limit");

const connectDB = require("./config/db");
const logger = require("./config/logger");
const errorHandler = require("./middleware/errorHandler");
const analysisRoutes = require("./routes/analysis");
const { checkHealth } = require("./services/pythonBridge");

const app = express();
const PORT = process.env.PORT || 5000;

// ---------------------------------------------------------------------------
// Security middleware
// ---------------------------------------------------------------------------

// Helmet: sets secure HTTP headers
app.use(helmet());

// CORS: allow React dev server and production frontend
app.use(
  cors({
    origin: [
      "http://localhost:5173",  // Vite dev server
      "http://localhost:3000",  // CRA dev server
      "https://quantmind.vercel.app", // Production
    ],
    methods: ["GET", "POST", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

// Rate limiting: 100 requests per 15 minutes per IP
const limiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000,
  max: parseInt(process.env.RATE_LIMIT_MAX) || 100,
  message: { error: "Too many requests. Please try again later." },
  standardHeaders: true,
  legacyHeaders: false,
});
app.use(limiter);

// ---------------------------------------------------------------------------
// Request parsing + logging
// ---------------------------------------------------------------------------

app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// Morgan HTTP request logging (skip in test)
if (process.env.NODE_ENV !== "test") {
  app.use(
    morgan("combined", {
      stream: { write: (msg) => logger.info(msg.trim()) },
    })
  );
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

// Health check
app.get("/health", async (req, res) => {
  const pythonHealthy = await checkHealth();
  res.json({
    status: "ok",
    server: "express",
    python_api: pythonHealthy ? "ok" : "unreachable",
    timestamp: new Date().toISOString(),
  });
});

// Analysis routes
app.use("/api/analysis", analysisRoutes);

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: `Route ${req.method} ${req.path} not found` });
});

// Global error handler (must be last)
app.use(errorHandler);

// ---------------------------------------------------------------------------
// Start server
// ---------------------------------------------------------------------------

const startServer = async () => {
  // Connect to MongoDB (non-blocking — server starts even if DB is down)
  await connectDB();

  app.listen(PORT, () => {
    logger.info(
      "QuantMind Express server running | port=%d | env=%s",
      PORT,
      process.env.NODE_ENV || "development"
    );
    logger.info("Python API URL: %s", process.env.PYTHON_API_URL || "http://localhost:8000");
  });
};

startServer();

module.exports = app; // Export for testing
