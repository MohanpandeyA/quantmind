/**
 * MongoDB connection using Mongoose.
 * Uses MongoDB Atlas free tier (512MB) — no cost.
 *
 * Connection is established once at startup and reused.
 * Mongoose handles connection pooling automatically.
 */

const mongoose = require("mongoose");
const logger = require("./logger");

/**
 * Connect to MongoDB Atlas.
 * Retries on failure with exponential backoff.
 *
 * @returns {Promise<void>}
 */
const connectDB = async () => {
  const uri = process.env.MONGODB_URI;

  if (!uri) {
    logger.warn(
      "MONGODB_URI not set — running without database. " +
        "Analysis history will not be saved. " +
        "Add MONGODB_URI to .env for persistence."
    );
    return;
  }

  try {
    await mongoose.connect(uri, {
      serverSelectionTimeoutMS: 5000,
      socketTimeoutMS: 45000,
    });

    logger.info("MongoDB connected | host=%s", mongoose.connection.host);

    // Handle connection events
    mongoose.connection.on("error", (err) => {
      logger.error("MongoDB connection error: %s", err.message);
    });

    mongoose.connection.on("disconnected", () => {
      logger.warn("MongoDB disconnected — attempting reconnect");
    });
  } catch (err) {
    logger.error("MongoDB connection failed: %s", err.message);
    logger.warn("Continuing without database — analysis history disabled");
  }
};

module.exports = connectDB;
