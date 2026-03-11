"""
utils/logger.py
==============
Structured, rotating logging system for the autonomous trading platform.
All agents use this unified logger for consistent, searchable output.
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """
    Emits log records as structured JSON lines for easy parsing and
    ingestion into observability stacks (e.g., ELK, Loki, CloudWatch).
    """

    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach extra structured context if present
        for key in ("agent", "strategy_id", "cycle", "phase", "metric"):
            val = getattr(record, key, None)
            if val is not None:
                base[key] = val

        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)

        return json.dumps(base, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable format for console output during development."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        agent = getattr(record, "agent", record.name)
        phase = getattr(record, "phase", "")
        phase_str = f"[{phase}] " if phase else ""
        msg = record.getMessage()
        formatted = f"{color}{record.levelname:8s}{self.RESET} | {agent:30s} | {phase_str}{msg}"
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        return formatted


def setup_logger(
    name: str,
    log_dir: str = "./logs",
    level: str = "INFO",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure and return a named logger with:
      - Rotating JSON file handler  (machine-parseable)
      - Console handler             (human-readable)

    Parameters
    ----------
    name         : Logger name (usually __name__ of the calling module)
    log_dir      : Directory where log files are written
    level        : Logging threshold ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    max_bytes    : Maximum size of each log file before rotation
    backup_count : Number of rotated files to keep
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        # Avoid adding duplicate handlers when the same module is imported
        # multiple times (common in async/multiprocessing contexts)
        return logger

    # ── JSON rotating file handler ──────────────────────────────────────
    file_path = os.path.join(log_dir, "trading_system.jsonl")
    file_handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(StructuredFormatter())

    # ── Console handler ─────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(HumanFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


class AgentLogger:
    """
    Convenience wrapper that binds agent-specific context to every log call.
    Usage:
        log = AgentLogger("StrategyInventor", config)
        log.info("Generated new strategy", phase="INVENTION", strategy_id="abc123")
    """

    def __init__(self, agent_name: str, config: Optional[Dict] = None):
        cfg = config or {}
        log_cfg = cfg.get("logging", {})
        self._logger = setup_logger(
            name=agent_name,
            log_dir=log_cfg.get("log_dir", "./logs"),
            level=log_cfg.get("level", "INFO"),
            max_bytes=log_cfg.get("max_bytes", 10_485_760),
            backup_count=log_cfg.get("backup_count", 5),
        )
        self._agent_name = agent_name

    def _log(self, level: str, msg: str, **kwargs):
        extra = {"agent": self._agent_name, **kwargs}
        getattr(self._logger, level)(msg, extra=extra)

    def debug(self, msg: str, **kwargs):   self._log("debug",   msg, **kwargs)
    def info(self, msg: str, **kwargs):    self._log("info",    msg, **kwargs)
    def warning(self, msg: str, **kwargs): self._log("warning", msg, **kwargs)
    def error(self, msg: str, **kwargs):   self._log("error",   msg, **kwargs)
    def critical(self, msg: str, **kwargs):self._log("critical", msg, **kwargs)

    def exception(self, msg: str, **kwargs):
        extra = {"agent": self._agent_name, **kwargs}
        self._logger.exception(msg, extra=extra)
