from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategies.pdh_acceptance_h4trend_v1 import (
    StrategyV1Config,
    find_signal_indices,
    map_prior_day_levels,
    merge_h4_context,
    simulate_trades,
)


def test_prior_day_levels_no_lookahead():
    d1 = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "high": [1.2, 1.3],
            "low": [1.1, 1.0],
        }
    )
    h1 = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2024-01-02 01:00", "2024-01-02 02:00"]
            ),
            "open": [1.21, 1.22],
            "high": [1.23, 1.24],
            "low": [1.20, 1.21],
            "close": [1.22, 1.23],
        }
    )
    mapped = map_prior_day_levels(h1, d1)
    assert mapped["pdh"].iloc[0] == 1.2
    assert mapped["pdl"].iloc[1] == 1.1


def test_merge_asof_backward_alignment():
    h4 = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 04:00"]),
            "ema50": [1.0, 2.0],
            "ema200": [1.0, 2.0],
            "slope": [0.1, 0.1],
            "trend_dir": ["LONG", "LONG"],
            "atr_pct": [50.0, 55.0],
            "vol_ok": [True, True],
        }
    )
    h1 = pd.DataFrame(
        {"time": pd.to_datetime(["2024-01-01 03:00", "2024-01-01 04:30"])}
    )
    merged = merge_h4_context(h1, h4)
    assert merged["ema50"].iloc[0] == 1.0
    assert merged["ema50"].iloc[1] == 2.0


def test_signal_requires_two_closes_beyond_pdh():
    h1 = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"]),
            "open": [1.0, 1.0],
            "high": [1.1, 1.2],
            "low": [0.9, 0.95],
            "close": [1.05, 0.99],
            "trend_dir": ["LONG", "LONG"],
            "vol_ok": [True, True],
            "pdh": [1.0, 1.0],
            "pdl": [0.9, 0.9],
        }
    )
    assert find_signal_indices(h1) == []

    h1.loc[1, "close"] = 1.02
    signals = find_signal_indices(h1)
    assert signals == [(0, "LONG")]


def test_intrabar_worst_case_sl_first():
    h1 = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00"]
            ),
            "open": [1.0, 1.001, 1.002],
            "high": [1.002, 1.004, 1.008],
            "low": [1.0, 1.001, 0.999],
            "close": [1.001, 1.002, 1.002],
            "trend_dir": ["LONG", "LONG", "LONG"],
            "vol_ok": [True, True, True],
            "pdh": [1.0, 1.0, 1.0],
            "pdl": [0.9, 0.9, 0.9],
            "atr_pct": [50.0, 50.0, 50.0],
            "ema50": [1.0, 1.0, 1.0],
            "ema200": [0.9, 0.9, 0.9],
            "slope": [0.1, 0.1, 0.1],
        }
    )
    trades = simulate_trades(h1, StrategyV1Config(quiet=True))
    assert len(trades) == 1
    trade = trades[0]
    assert trade["exit_reason"] == "SL"
    assert trade["exit_price"] == trade["sl"]
