"""Indicator functions used in tests and utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range - optimized with pandas EWM."""
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ),
    )
    tr[0] = 0
    alpha = 1.0 / period
    return pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values


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
