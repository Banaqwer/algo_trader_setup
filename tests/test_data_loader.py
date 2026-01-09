import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_loader import _resample_ohlc
from utils.timeframes import merge_higher_timeframes


def test_resample_ohlc_correctness():
    idx = pd.date_range(
        "2025-01-01 00:05", periods=12, freq="5min", tz="UTC", name="timestamp"
    )
    df = pd.DataFrame(
        {
            "open": list(range(1, 13)),
            "high": [value + 2 for value in range(101, 113)],
            "low": [value - 1 for value in range(1, 13)],
            "close": list(range(101, 113)),
        },
        index=idx,
    )

    resampled = _resample_ohlc(df, "1h")

    assert len(resampled) == 1
    row = resampled.iloc[0]
    assert row["open"] == 1
    assert row["close"] == 112
    assert row["high"] == max(df["high"])
    assert row["low"] == min(df["low"])


def test_resample_has_no_duplicate_timestamps():
    idx = pd.date_range(
        "2025-01-01 00:05", periods=12, freq="5min", tz="UTC", name="timestamp"
    )
    df = pd.DataFrame(
        {
            "open": range(12),
            "high": range(1, 13),
            "low": range(12),
            "close": range(12),
        },
        index=idx,
    )
    duplicate = df.iloc[[0]].copy()
    duplicate.index = [idx[0]]
    df = pd.concat([df, duplicate])

    resampled = _resample_ohlc(df, "1h")

    assert resampled.index.is_unique


def test_merge_asof_backward_alignment():
    h1_index = pd.DatetimeIndex(
        ["2025-01-01 01:00", "2025-01-01 05:00"], tz="UTC", name="timestamp"
    )
    h1 = pd.DataFrame({"open": [1.0, 2.0]}, index=h1_index)

    h4_index = pd.DatetimeIndex(
        ["2025-01-01 04:00"], tz="UTC", name="timestamp"
    )
    h4 = pd.DataFrame({"open": [10.0]}, index=h4_index)

    d1_index = pd.DatetimeIndex(
        ["2025-01-01 00:00"], tz="UTC", name="timestamp"
    )
    d1 = pd.DataFrame({"open": [100.0]}, index=d1_index)

    merged = merge_higher_timeframes(h1, h4, d1)

    assert pd.isna(merged.loc[pd.Timestamp("2025-01-01 01:00", tz="UTC"), "open_H4"])
    assert (
        merged.loc[pd.Timestamp("2025-01-01 05:00", tz="UTC"), "open_H4"] == 10.0
    )
    assert merged.loc[pd.Timestamp("2025-01-01 01:00", tz="UTC"), "open_D1"] == 100.0
