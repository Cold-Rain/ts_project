from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.models.base import BaseForecastModel


class ARIMAModel(BaseForecastModel):
    """ARIMA model using statsmodels SARIMAX backend."""
    name = "ARIMA"

    def __init__(self, order: tuple[int, int, int] = (5, 1, 0)) -> None:
        self.order = order
        self.model_fit = None

    def fit(self, series: pd.Series) -> None:
        if series.index.freq is None:
            series = series.asfreq("B").ffill().bfill()

        best_aic = np.inf
        best_order: tuple[int, int, int] | None = None
        best_fit = None

        for p in range(0, 4):
            for q in range(0, 4):
                order = (p, 1, q)
                try:
                    model = SARIMAX(
                        series,
                        order=order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(disp=False)
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = order
                        best_fit = fitted
                except Exception:
                    continue

        if best_fit is None or best_order is None:
            fallback = (1, 1, 1)
            model = SARIMAX(
                series,
                order=fallback,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model_fit = model.fit(disp=False)
            self.order = fallback
            self.name = f"ARIMA{self.order}"
            return

        self.model_fit = best_fit
        self.order = best_order
        self.name = f"ARIMA{self.order}"

    def predict(self, horizon: int) -> np.ndarray:
        if self.model_fit is None:
            raise RuntimeError("Model must be fit before prediction.")

        forecast = self.model_fit.forecast(steps=horizon)
        return np.asarray(forecast)
