from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class Signal:
    date: pd.Timestamp
    action: str
    price: float


def _smooth_series(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def generate_signals(
    forecast: pd.Series, order: int = 2, smooth_window: int = 3
) -> List[Signal]:
    """Detect local minima/maxima on the forecast series and emit BUY/SELL signals."""
    if len(forecast) < (order * 2 + 1):
        return []

    smoothed = _smooth_series(forecast, smooth_window)
    signals: List[Signal] = []

    for idx in range(order, len(smoothed) - order):
        window = smoothed.iloc[idx - order : idx + order + 1]
        value = smoothed.iloc[idx]
        if window.min() == window.max():
            continue

        date = forecast.index[idx]
        price = float(forecast.iloc[idx])
        if value == window.min():
            signals.append(Signal(date=date, action="BUY", price=price))
        elif value == window.max():
            signals.append(Signal(date=date, action="SELL", price=price))

    return signals


def simulate_strategy(
    amount: float, forecast: pd.Series, signals: List[Signal]
) -> dict:
    """Simulate a simple long-only strategy using forecast signals."""
    cash = float(amount)
    shares = 0.0
    trades = []

    signal_map = {sig.date: sig for sig in signals}
    for date, price in forecast.items():
        signal = signal_map.get(date)
        if signal is None:
            continue

        if signal.action == "BUY" and shares == 0.0:
            shares = cash / price
            cash = 0.0
            trades.append((date, "BUY", price))
        elif signal.action == "SELL" and shares > 0.0:
            cash = shares * price
            shares = 0.0
            trades.append((date, "SELL", price))

    if shares > 0.0:
        last_price = float(forecast.iloc[-1])
        cash = shares * last_price
        shares = 0.0
        trades.append((forecast.index[-1], "SELL", last_price))

    profit_abs = cash - float(amount)
    profit_pct = (profit_abs / float(amount)) * 100.0 if amount else 0.0

    return {
        "profit_abs": profit_abs,
        "profit_pct": profit_pct,
        "final_value": cash,
        "trades": trades,
    }
