from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

DATA_PERIOD = "2y"
DATA_INTERVAL = "1d"
MIN_SERIES_POINTS = 200
TEST_SIZE = 60
FORECAST_HORIZON = 30

LAG_FEATURES = 10
LSTM_WINDOW = 20
LSTM_EPOCHS = 30
LSTM_BATCH_SIZE = 16
LSTM_PATIENCE = 5
LSTM_VAL_SPLIT = 0.1
LSTM_VAL_MIN = 20
LSTM_DROPOUT = 0.1

PLOT_HISTORY_DAYS = 365

WALK_FORWARD_FOLDS = 3

LOG_DIR = BASE_DIR / "logs"
REQUESTS_CSV = LOG_DIR / "requests.csv"
APP_LOG = LOG_DIR / "app.log"

DATE_FORMAT = "%Y-%m-%d"
