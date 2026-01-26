"""Indicator functions used in tests and utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR) with Wilder smoothing."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr_components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    tr.iloc[0] = np.nan

    atr_values = pd.Series(index=df.index, dtype=float)
    if len(tr) <= period:
        return atr_values

    initial_window = tr.iloc[1 : period + 1]
    initial_atr = initial_window.mean(skipna=True)
    if pd.isna(initial_atr):
        return atr_values

    atr_values.iloc[:period] = np.nan
    atr_values.iloc[period] = initial_atr
    for i in range(period + 1, len(tr)):
        atr_values.iloc[i] = (atr_values.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period

    return atr_values


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average - optimized with pandas EWM."""
    alpha = 2 / (period + 1)
    return pd.Series(data).ewm(alpha=alpha, adjust=False).mean().values


def rolling_percentile(values: np.ndarray, window: int = 100) -> np.ndarray:
    """Rolling percentile with a fixed default for warm-up values."""
    values = np.asarray(values)
    percentiles = np.full(len(values), 50.0)
    for i in range(window, len(values)):
        window_values = values[i - window : i + 1]
        percentiles[i] = stats.percentileofscore(window_values, values[i])
    return percentiles
