/**
 * Winston structured logger for the QuantMind Express server.
 * Outputs JSON in production, colorized text in development.
 */

const { createLogger, format, transports } = require("winston");

const isDev = process.env.NODE_ENV !== "production";

const logger = createLogger({
  level: isDev ? "debug" : "info",
  format: isDev
    ? format.combine(
        format.colorize(),
        format.timestamp({ format: "HH:mm:ss" }),
        format.printf(({ timestamp, level, message, ...meta }) => {
          const metaStr = Object.keys(meta).length
            ? " " + JSON.stringify(meta)
            : "";
          return `${timestamp} ${level}: ${message}${metaStr}`;
        })
      )
    : format.combine(format.timestamp(), format.json()),
  transports: [new transports.Console()],
});

module.exports = logger;
