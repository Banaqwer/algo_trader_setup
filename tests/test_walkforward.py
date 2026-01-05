import pandas as pd

from walkforward import generate_walkforward_windows


def test_walkforward_purge_and_embargo_no_overlap():
    start = pd.Timestamp("2025-01-01")
    end = pd.Timestamp("2026-06-30")

    windows = generate_walkforward_windows(
        start_date=start,
        end_date=end,
        train_years=1,
        test_months=2,
        step_months=3,
        purge_days=5,
        embargo_days=2,
    )

    assert windows, "Expected at least one walk-forward window"

    for w in windows:
        # train end precedes purge end and embargo end
        assert w.train_end < w.purge_end
        assert w.purge_end <= w.embargo_end
        # embargo should precede test start
        assert w.embargo_end <= w.test_start
        # train period should not overlap test period
        assert w.train_end < w.test_start
        assert w.test_start <= w.test_end

    # ensure windows are non-overlapping in test segments
    for prev, nxt in zip(windows, windows[1:]):
        assert prev.test_end < nxt.test_start
