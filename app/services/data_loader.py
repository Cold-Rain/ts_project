from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf

from app.config import DATA_INTERVAL, DATA_PERIOD, MIN_SERIES_POINTS


def _download_primary(ticker: str) -> pd.DataFrame:
    return yf.download(
        tickers=ticker,
        period=DATA_PERIOD,
        interval=DATA_INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=20,
    )


def _download_fallback(ticker: str) -> pd.DataFrame:
    ticker_obj = yf.Ticker(ticker)
    return ticker_obj.history(period=DATA_PERIOD, interval=DATA_INTERVAL, auto_adjust=False)


def load_price_series(ticker: str, logger: logging.Logger | None = None) -> pd.Series:
    """Download and validate the daily price series for a ticker."""
    data = pd.DataFrame()
    try:
        data = _download_primary(ticker)
    except Exception as exc:
        if logger:
            logger.error("yfinance download failed for %s: %s", ticker, exc)
        data = pd.DataFrame()

    if data is None or data.empty:
        if logger:
            logger.warning("Primary download empty for %s, trying fallback", ticker)
        try:
            time.sleep(1)
            data = _download_fallback(ticker)
        except Exception as exc:
            if logger:
                logger.error("yfinance fallback failed for %s: %s", ticker, exc)
            data = pd.DataFrame()

    if data is None or data.empty:
        raise RuntimeError(
            f"No data returned for '{ticker}'. Check the symbol or try again later."
        )

    series: pd.Series | pd.DataFrame
    if isinstance(data.columns, pd.MultiIndex):
        fields = data.columns.get_level_values(0)
        if "Adj Close" in fields:
            series = data["Adj Close"]
        elif "Close" in fields:
            series = data["Close"]
        else:
            raise ValueError("No usable price column in data.")
    else:
        if "Adj Close" in data.columns:
            series = data["Adj Close"]
        elif "Close" in data.columns:
            series = data["Close"]
        else:
            raise ValueError("No usable price column in data.")

    if isinstance(series, pd.DataFrame):
        if ticker in series.columns:
            series = series[ticker]
        else:
            series = series.iloc[:, 0]

    series = series.dropna().sort_index()
    series.index = pd.to_datetime(series.index)

    full_index = pd.bdate_range(series.index.min(), series.index.max())
    series = series.reindex(full_index).ffill().bfill()

    if len(series) < MIN_SERIES_POINTS:
        raise ValueError(
            f"Not enough data points ({len(series)}). Need at least {MIN_SERIES_POINTS}."
        )

    return series
