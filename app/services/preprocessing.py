from __future__ import annotations

import pandas as pd

from app.config import TEST_SIZE, WALK_FORWARD_FOLDS


def train_test_split_series(
    series: pd.Series, test_size: int = TEST_SIZE
) -> tuple[pd.Series, pd.Series]:
    """Split series into train and test portions with a fixed test size."""
    if len(series) <= test_size:
        raise ValueError("Series too short for train/test split.")

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test


def generate_walk_forward_folds(
    series: pd.Series,
    test_size: int = TEST_SIZE,
    n_folds: int = WALK_FORWARD_FOLDS,
) -> list[tuple[pd.Series, pd.Series]]:
    """Generate walk-forward folds for time-series evaluation."""
    total = len(series)
    if total <= test_size:
        raise ValueError("Series too short for walk-forward evaluation.")

    max_folds = (total - test_size) // test_size
    folds_count = max(1, min(n_folds, max_folds))

    folds: list[tuple[pd.Series, pd.Series]] = []
    for idx in range(folds_count):
        test_end = total - (folds_count - idx - 1) * test_size
        test_start = test_end - test_size
        train_end = test_start
        if train_end <= 0:
            continue
        train = series.iloc[:train_end]
        test = series.iloc[test_start:test_end]
        if len(train) == 0 or len(test) == 0:
            continue
        folds.append((train, test))

    if not folds:
        raise ValueError("Unable to generate walk-forward folds.")

    return folds
