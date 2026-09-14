from __future__ import annotations

import pandas as pd


def merge_asof_on_date(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    direction: str = "backward",
    suffixes: tuple[str, str] = ("", "_right"),
) -> pd.DataFrame:
    if "date" not in left.columns or "date" not in right.columns:
        raise ValueError("Both frames must have a 'date' column")

    left_sorted = left.sort_values("date")
    right_sorted = right.sort_values("date")
    return pd.merge_asof(
        left_sorted,
        right_sorted,
        on="date",
        direction=direction,
        suffixes=suffixes,
    )
