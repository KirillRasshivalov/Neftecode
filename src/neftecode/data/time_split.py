from __future__ import annotations

from datetime import datetime

import pandas as pd


def time_based_split(
    frame: pd.DataFrame,
    *,
    train_end: datetime,
    val_end: datetime,
    time_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if time_col not in frame.columns:
        raise ValueError(f"Missing time column '{time_col}'")

    ts = pd.to_datetime(frame[time_col])
    train = frame.loc[ts <= train_end]
    val = frame.loc[(ts > train_end) & (ts <= val_end)]
    test = frame.loc[ts > val_end]
    return train, val, test
