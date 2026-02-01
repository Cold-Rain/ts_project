from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

from app.config import (
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_PATIENCE,
    LSTM_VAL_MIN,
    LSTM_VAL_SPLIT,
    LSTM_WINDOW,
)
from app.models.base import BaseForecastModel


class LSTMForecastModel(BaseForecastModel):
    """Minimal LSTM forecaster with recursive multi-step prediction."""
    name = "LSTM"

    def __init__(
        self,
        window: int = LSTM_WINDOW,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH_SIZE,
        patience: int = LSTM_PATIENCE,
    ) -> None:
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.scaler = MinMaxScaler()
        self.model: Sequential | None = None
        self.last_window: np.ndarray | None = None
        self.last_price: float | None = None

    def _create_sequences(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = []
        y = []
        for idx in range(self.window, len(values)):
            X.append(values[idx - self.window : idx])
            y.append(values[idx])
        X_arr = np.asarray(X).reshape(-1, self.window, 1)
        y_arr = np.asarray(y)
        return X_arr, y_arr

    def fit(self, series: pd.Series) -> None:
        prices = series.values.astype(float).reshape(-1)
        if np.any(prices <= 0):
            raise ValueError("Prices must be positive for log-returns.")

        returns = np.diff(np.log(prices))
        if len(returns) <= self.window:
            raise ValueError("Series too short for LSTM window.")

        tf.keras.backend.clear_session()
        scaled = self.scaler.fit_transform(returns.reshape(-1, 1)).flatten()
        X, y = self._create_sequences(scaled)

        model = Sequential(
            [
                Input(shape=(self.window, 1)),
                LSTM(16),
                Dropout(LSTM_DROPOUT),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
            )
        ]

        val_size = max(1, int(len(X) * LSTM_VAL_SPLIT), LSTM_VAL_MIN)
        val_size = min(val_size, len(X) - 1) if len(X) > 1 else 1
        if val_size >= len(X):
            val_size = max(1, len(X) - 1)

        if len(X) > val_size:
            X_train, y_train = X[:-val_size], y[:-val_size]
            X_val, y_val = X[-val_size:], y[-val_size:]
            model.fit(
                X_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                shuffle=False,
                verbose=0,
            )
        else:
            model.fit(
                X,
                y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=False,
                verbose=0,
            )

        self.model = model
        self.last_window = scaled[-self.window :].copy()
        self.last_price = float(prices[-1])

    def predict(self, horizon: int) -> np.ndarray:
        if self.model is None or self.last_window is None or self.last_price is None:
            raise RuntimeError("Model must be fit before prediction.")

        window = self.last_window.copy()
        preds_scaled: list[float] = []
        for _ in range(horizon):
            X = window[-self.window :].reshape(1, self.window, 1)
            pred = float(self.model.predict(X, verbose=0)[0][0])
            preds_scaled.append(pred)
            window = np.append(window, pred)

        pred_returns = self.scaler.inverse_transform(
            np.asarray(preds_scaled).reshape(-1, 1)
        ).flatten()
        pred_prices = self.last_price * np.exp(np.cumsum(pred_returns))
        return pred_prices
