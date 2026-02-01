from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.config import PLOT_HISTORY_DAYS
from app.services.recommendations import Signal


def plot_forecast(
    ticker: str,
    history: pd.Series,
    forecast: pd.Series,
    signals: list[Signal],
    model_name: str,
) -> BytesIO:
    history_plot = history.tail(PLOT_HISTORY_DAYS)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history_plot.index, history_plot.values, label="History")
    ax.plot(forecast.index, forecast.values, label="Forecast")

    for signal in signals:
        if signal.action == "BUY":
            ax.scatter(signal.date, signal.price, marker="^", color="green", label="BUY")
        elif signal.action == "SELL":
            ax.scatter(signal.date, signal.price, marker="v", color="red", label="SELL")

    ax.set_title(f"{ticker} | {model_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best")

    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)
    return buffer
