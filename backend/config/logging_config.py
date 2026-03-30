"""Structured logging configuration for QuantMind backend.

Sets up a consistent logging format across all modules.
For live trading, uses an async QueueHandler so logging never
blocks the signal-generation hot path.

Import and call `setup_logging()` once at application startup.
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
import sys
import threading
from typing import Optional

from config.settings import settings

# Global async log queue — shared across all loggers
_log_queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=10_000)
_queue_listener: Optional[logging.handlers.QueueListener] = None


def setup_logging(level: Optional[str] = None, async_mode: bool = True) -> None:
    """Configure root logger with structured format.

    In async_mode (default for live trading):
        - Log records are placed on a queue (non-blocking, ~50ns)
        - A background thread drains the queue and writes to stdout
        - The trading hot path is NEVER blocked by I/O

    In sync_mode (for testing/debugging):
        - Log records are written directly to stdout (blocking)

    Args:
        level: Override log level. Defaults to settings.log_level.
        async_mode: If True, use non-blocking async queue handler.
                    Set False for unit tests to avoid thread leaks.
    """
    global _queue_listener

    log_level = getattr(logging, (level or settings.log_level).upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates on re-import
    root_logger.handlers.clear()

    if async_mode:
        # Non-blocking queue handler — puts record on queue in ~50ns
        queue_handler = logging.handlers.QueueHandler(_log_queue)
        root_logger.addHandler(queue_handler)

        # Background thread drains queue and writes to stdout
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        if _queue_listener is not None:
            _queue_listener.stop()

        _queue_listener = logging.handlers.QueueListener(
            _log_queue,
            stream_handler,
            respect_handler_level=True,
        )
        _queue_listener.start()
    else:
        # Synchronous mode — for tests and debugging
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def stop_logging() -> None:
    """Stop the async queue listener. Call at application shutdown.

    Important: Always call this before exiting to flush remaining
    log records from the queue.
    """
    global _queue_listener
    if _queue_listener is not None:
        _queue_listener.stop()
        _queue_listener = None


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Backtester initialized")
    """
    return logging.getLogger(name)
