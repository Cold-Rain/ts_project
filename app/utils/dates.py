from __future__ import annotations

import pandas as pd


def future_business_dates(start_date: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    next_day = start_date + pd.offsets.BDay(1)
    return pd.bdate_range(start=next_day, periods=periods)
