"""walkforward.py - walk-forward split utilities with purge and embargo."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import pandas as pd


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    purge_end: pd.Timestamp
    embargo_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_walkforward_windows(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_years: int,
    test_months: int,
    step_months: int,
    purge_days: int,
    embargo_days: int,
) -> List[WalkForwardWindow]:
    """
    Generate walk-forward windows with purge + embargo to prevent leakage.

    Returns a list of WalkForwardWindow entries. Windows stop when the next
    train/test period would extend beyond `end_date`.
    """

    windows: List[WalkForwardWindow] = []
    current_train_start = pd.Timestamp(start_date)

    while True:
        train_end = current_train_start + pd.DateOffset(years=train_years) - timedelta(days=1)
        purge_end = train_end + timedelta(days=purge_days)
        test_start = purge_end
        embargo_end = test_start + timedelta(days=embargo_days)
        test_start = embargo_end
        test_end = test_start + pd.DateOffset(months=test_months) - timedelta(days=1)

        if test_end > end_date:
            break

        windows.append(
            WalkForwardWindow(
                train_start=current_train_start,
                train_end=train_end,
                purge_end=purge_end,
                embargo_end=embargo_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        # advance by step_months from current train start
        current_train_start = current_train_start + pd.DateOffset(months=step_months)
        if current_train_start >= end_date:
            break

    return windows
