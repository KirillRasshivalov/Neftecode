from datetime import datetime

from neftecode.data.sync import merge_asof_on_date
from neftecode.data.time_split import time_based_split
import pandas as pd


def test_merge_asof_uses_time_not_row_order():
    left = pd.DataFrame({"date": pd.to_datetime(["2024-01-01 00:20", "2024-01-01 00:00"]), "x": [2, 1]})
    right = pd.DataFrame({"date": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:10"]), "q": [10, 20]})
    out = merge_asof_on_date(left, right)
    assert list(out.sort_values("date")["q"]) == [10, 20]


def test_time_based_split_no_shuffle():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
            "v": [1, 2, 3, 4],
        }
    )
    train, val, test = time_based_split(
        frame,
        train_end=datetime(2024, 2, 1),
        val_end=datetime(2024, 3, 1),
    )
    assert list(train["v"]) == [1, 2]
    assert list(val["v"]) == [3]
    assert list(test["v"]) == [4]
