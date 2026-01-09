"""Timeframe utilities for resampling and alignment."""

from __future__ import annotations

import pandas as pd

TIMEFRAME_RULES = {
    "M5": "5min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}

EXPECTED_DELTAS = {
    timeframe: pd.to_timedelta(rule)
    for timeframe, rule in TIMEFRAME_RULES.items()
}


def expected_timedelta(timeframe: str) -> pd.Timedelta:
    """Return the expected timedelta for a timeframe label."""
    if timeframe not in EXPECTED_DELTAS:
        raise KeyError(f"Unknown timeframe: {timeframe}")
    return EXPECTED_DELTAS[timeframe]


def merge_asof_backward(
    base: pd.DataFrame, context: pd.DataFrame, suffix: str
) -> pd.DataFrame:
    """Merge context data into base using backward-looking asof alignment."""
    base_sorted = base.sort_index()
    context_sorted = context.sort_index()
    context_sorted = context_sorted.rename(
        columns={col: f"{col}_{suffix}" for col in context_sorted.columns}
    )

    merged = pd.merge_asof(
        base_sorted.reset_index(),
        context_sorted.reset_index(),
        on="timestamp",
        direction="backward",
    )
    return merged.set_index("timestamp")


def merge_higher_timeframes(
    h1: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame
) -> pd.DataFrame:
    """Attach H4 and D1 context to H1 timestamps with backward-only alignment."""
    merged = merge_asof_backward(h1, h4, "H4")
    merged = merge_asof_backward(merged, d1, "D1")
    return merged
