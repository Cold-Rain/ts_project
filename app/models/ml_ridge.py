from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from app.config import LAG_FEATURES
from app.models.base import BaseForecastModel


class RidgeLagModel(BaseForecastModel):
    """Ridge regression on lagged values with recursive forecasting."""
    name = "RidgeLag"

    def __init__(self, lag: int = LAG_FEATURES, alpha: float = 1.0) -> None:
        self.lag = lag
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)
        self.scaler = StandardScaler()
        self.last_window: np.ndarray | None = None
        self.last_price: float | None = None

    def _create_lag_matrix(
        self, series: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        prices = np.asarray(series, dtype=float).reshape(-1)
        if np.any(prices <= 0):
            raise ValueError("Prices must be positive for log-returns.")

        returns = np.diff(np.log(prices))
        if len(returns) <= self.lag:
            raise ValueError("Series too short for lag features.")

        X = []
        y = []
        for idx in range(self.lag, len(returns)):
            X.append(returns[idx - self.lag : idx])
            y.append(returns[idx])
        return np.asarray(X), np.asarray(y), returns, prices

    def fit(self, series: pd.Series) -> None:
        X, y, returns, prices = self._create_lag_matrix(series)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.last_window = returns[-self.lag :].astype(float)
        self.last_price = float(prices[-1])

    def predict(self, horizon: int) -> np.ndarray:
        if self.last_window is None or self.last_price is None:
            raise RuntimeError("Model must be fit before prediction.")

        window = self.last_window.copy()
        pred_returns: list[float] = []
        for _ in range(horizon):
            X = window[-self.lag :].reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            pred = float(self.model.predict(X_scaled)[0])
            pred_returns.append(pred)
            window = np.append(window, pred)
        pred_returns_arr = np.asarray(pred_returns)
        pred_prices = self.last_price * np.exp(np.cumsum(pred_returns_arr))
        return pred_prices
