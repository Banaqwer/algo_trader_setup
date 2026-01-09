import numpy as np
import pandas as pd

from indicators import atr, ema, rolling_percentile


def test_ema_small_sequence():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ema(series, period=3)

    assert result.iloc[:2].isna().all()

    expected = pd.Series([np.nan, np.nan, 2.25, 3.125, 4.0625])
    pd.testing.assert_series_equal(result, expected)


def test_atr_sanity_synthetic_ohlc():
    df = pd.DataFrame(
        {
            "open": [0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            "high": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "low": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "close": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )
    result = atr(df, period=3)

    assert result.iloc[:3].isna().all()
    assert result.dropna().iloc[-1] == 1.0


def test_rolling_percentile_properties():
    series = pd.Series(range(1, 11))
    result = rolling_percentile(series, lookback=5)

    assert result.iloc[:4].isna().all()
    assert (result.dropna() == 100.0).all()

    mixed = pd.Series([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
    mixed_result = rolling_percentile(mixed, lookback=4)

    assert mixed_result.dropna().between(0, 100).all()
