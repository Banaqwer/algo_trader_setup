"""Technical indicators for Strategy V1."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Compute an exponential moving average with standard EMA smoothing.

    Uses alpha = 2 / (period + 1) with no lookahead and NaNs until warmup.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR) with Wilder smoothing.

    True Range uses the previous close. ATR uses Wilder's EMA
    (alpha = 1 / period). Output is NaN until warmup is available.
    """
    if period <= 0:
        raise ValueError("period must be positive")

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rolling_percentile(series: pd.Series, lookback: int) -> pd.Series:
    """Return percentile rank (0..100) over a trailing lookback window."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")

    def _percentile(window: np.ndarray) -> float:
        current = window[-1]
        n = len(window)
        if n == 1:
            return 100.0
        less = np.sum(window < current)
        equal = np.sum(window == current)
        rank = less + (equal - 1) / 2
        return (rank / (n - 1)) * 100

    return series.rolling(window=lookback, min_periods=lookback).apply(
        _percentile, raw=True
    )
