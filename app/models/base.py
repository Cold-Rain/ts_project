from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseForecastModel(ABC):
    """Interface for time-series forecasting models."""
    name: str

    @abstractmethod
    def fit(self, series: pd.Series) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        raise NotImplementedError
