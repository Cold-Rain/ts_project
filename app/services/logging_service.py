from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.config import APP_LOG, LOG_DIR, REQUESTS_CSV

LOGGER_NAME = "session_bot"
REQUEST_FIELDS = [
    "user_id",
    "timestamp",
    "ticker",
    "amount",
    "best_model",
    "metric_name",
    "metric_value",
    "estimated_profit",
    "current_price",
    "forecast_end_price",
    "change_pct",
]


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(APP_LOG, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def _ensure_requests_header(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=REQUEST_FIELDS)
            writer.writeheader()


def log_request(row: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_requests_header(REQUESTS_CSV)

    full_row = {key: row.get(key, "") for key in REQUEST_FIELDS}
    if not full_row["timestamp"]:
        full_row["timestamp"] = datetime.now(timezone.utc).isoformat()

    with REQUESTS_CSV.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REQUEST_FIELDS)
        writer.writerow(full_row)
